
-- mandra_kernels.hs - Mandra Atomic Risk Governance
-- Level 1-4 Risk Gates | GP Variance | Kelly Fusion
-- Production Risk Engine | SOS-27-X + Adelic Tube Native

import qualified Data.Aeson as A
import qualified Data.ByteString.Lazy as B
import qualified Data.Deque as D
import Control.Monad (forever)
import Control.Concurrent (threadDelay)
import Control.Monad.IO.Class (liftIO)
import Control.Concurrent.STM
import Data.Maybe (fromMaybe)
import Data.List (genericLength)
import Control.Exception (catch, SomeException)

-- CPU-First Config
-- Note: Haskell does not have a direct equivalent for JAX configurations.

data MandraSignal = MandraSignal {
    size :: Double,
    stopDistance :: Double,
    level :: Int,              -- 1-4 active gates
    kellyFraction :: Double,
    gpVariance :: Double,
    edgePerVar :: Double,
    positionLimit :: Int
} deriving (Show)

data MandraKernels = MandraKernels {
    equity :: Double,
    maxPositions :: Int,
    balanceHistory :: D.Deque Double,
    positionHistory :: D.Deque Double,
    drawdownHistory :: D.Deque Double,
    alpha :: Double,
    lengthScale :: Double,
    level4Threshold :: Double,
    level2AtrMult :: Double
}

-- Constructor for MandraKernels
createMandraKernels :: Double -> MandraKernels
createMandraKernels equity = MandraKernels {
    equity = equity,
    maxPositions = 3,
    balanceHistory = D.empty,
    positionHistory = D.empty,
    drawdownHistory = D.empty,
    alpha = 1e-6,
    lengthScale = 0.1,
    level4Threshold = 0.12,
    level2AtrMult = 2.0
}

-- ========================================
-- LEVEL 4: GLOBAL CIRCUIT BREAKER
-- ========================================
checkLevel4 :: [Double] -> Bool
checkLevel4 balanceHistory = drawdownPct > level4Threshold
  where
    peak = maximum balanceHistory
    current = last balanceHistory
    drawdownPct = (peak - current) / peak

-- ========================================
-- LEVEL 3: CONCURRENCY LIMIT
-- ========================================
checkLevel3 :: STM Int
checkLevel3 = do
    positions <- getPositionsFromRedis
    return $ length positions

-- ========================================
-- LEVEL 2: VOLATILITY GATE
-- ========================================
checkLevel2 :: Double -> Double -> Bool
checkLevel2 atrCurrent atrEma20 = atrCurrent > level2AtrMult * atrEma20

-- ========================================
-- LEVEL 1: KELLY + GP VARIANCE
-- ========================================
kellyCriterion :: Double -> Double -> Double -> Double
kellyCriterion edge odds gpVar = clip (baseKelly * variancePenalty) 0.0 0.25
  where
    baseKelly = (edge * odds - 1) / odds
    variancePenalty = 1.0 / sqrt (gpVar + 1e-8)

gaussianProcessVariance :: [Double] -> Double
gaussianProcessVariance returns
    | length returns < 10 = 1.0
    | otherwise = varPred
  where
    n = length returns
    x = [0..fromIntegral (n-1)]
    y = returns
    k = exp (-0.5 * ((x - transpose x) / lengthScale) ** 2)
    kY = k + alpha * identityMatrix n
    kS = exp (-0.5 * ((last x - x) / lengthScale) ** 2)
    varPred = 1.0 - kS `dot` (inverse kY) `dot` transpose kS

-- ========================================
-- EDGE PER VARIANCE RATIO
-- ========================================
edgePerVariance :: [Double] -> [Double] -> Double
edgePerVariance returns signals = edge / (variance + 1e-8)
  where
    edge = mean (zipWith (*) returns signals)
    variance = variance returns

-- ========================================
-- ATOMIC SIZING KERNEL
-- ========================================
atomicSizeKernel :: Double -> Double -> Double -> Double -> Double
atomicSizeKernel confidence atr edgePV gpVar = clip pipSize 0.0 10.0
  where
    kellyFrac = kellyCriterion edgePV 3.0 gpVar
    confMult = 0.5 + 1.5 * confidence
    riskBudget = equity * 0.01 * kellyFrac * confMult
    pipSize = riskBudget / (atr * 10000)

-- ========================================
-- SCALE-OUT KERNEL (30/40/30)
-- ========================================
scaleOutKernel :: Double -> Double -> Double -> [Double]
scaleOutKernel entryPrice currentPrice rrTarget = scaleOut ++ [trailing]
  where
    profitPips = (currentPrice - entryPrice) * 10000
    targets = [2.0 * rrTarget, 4.0 * rrTarget]
    scaleOut = [if profitPips >= target then 0.3 else 0.0 | target <- targets]
    trailing = max 0.0 (profitPips - 1.5 * rrTarget) * 0.4

-- ========================================
-- FULL MANDRA GOVERNANCE
-- ========================================
mandraGovern :: MandraKernels -> A.Value -> STM MandraSignal
mandraGovern mandra sosSignal = do
    tick <- getMarketStateFromRedis
    returns <- getReturnsFromRedis
    let level4 = checkLevel4 (D.toList (balanceHistory mandra))
    level3Count <- checkLevel3
    let level2 = checkLevel2 (fromMaybe 0.001 (A.decode (A.Object tick) >>= A.lookup "atr")) (mean (take 20 returns) * 0.02)
    let activeLevel = if level4 then 4 else if level3Count >= maxPositions mandra then 3 else if level2 then 2 else 1

    if level4
        then return $ MandraSignal 0.0 0.0 4 0.0 0.0 0.0 0
        else do
            let gpVar = gaussianProcessVariance (zipWith (-) (tail returns) returns)
            let edgePV = edgePerVariance (zipWith (-) (tail returns) returns) (replicate (length returns - 1) (A.decode (A.Object sosSignal) >>= A.lookup "confidence"))
            let size = atomicSizeKernel (A.decode (A.Object sosSignal) >>= A.lookup "confidence") (fromMaybe 0.001 (A.decode (A.Object tick) >>= A.lookup "atr")) edgePV gpVar
            let adjustedSize = size * fromIntegral (max 0 (maxPositions mandra - level3Count))
            return $ MandraSignal adjustedSize (fromMaybe 0.001 (A.decode (A.Object tick) >>= A.lookup "atr")) activeLevel edgePV gpVar edgePV (maxPositions mandra - level3Count)

-- ========================================
-- PRODUCTION RISK LOOP
-- ========================================
riskGovernanceLoop :: MandraKernels -> STM ()
riskGovernanceLoop mandra = forever $ do
    msgs <- getMessagesFromRedis
    case msgs of
        [] -> threadDelay 100000
        sosSignal:_ -> do
            mandraSignal <- mandraGovern mandra sosSignal
            let finalSignal = A.object [
                    "action" A..= (A.decode (A.Object sosSignal) >>= A.lookup "action"),
                    "symbol" A..= (A.decode (A.Object sosSignal) >>= A.lookup "symbol"),
                    "mandra_size" A..= size mandraSignal,
                    "stop_distance" A..= stopDistance mandraSignal,
                    "risk_level" A..= level mandraSignal,
                    "kelly_fraction" A..= kellyFraction mandraSignal,
                    "gates" A..= A.object [
                        "level_4_dd" A..= (level mandraSignal == 4),
                        "level_3_concurrency" A..= (level mandraSignal == 3),
                        "level_2_volatility" A..= (level mandraSignal == 2)
                    ],
                    "timestamp" A..= currentTime
                ]
            publishToRedis finalSignal
            let status = if level mandraSignal >= 4 then " HALTED" else "✅ SIZE=" ++ show (size mandraSignal)
            liftIO $ putStrLn $ show (A.decode (A.Object sosSignal) >>= A.lookup "symbol") ++ " MANDRA L" ++ show (level mandraSignal) ++ status ++ " | Kelly=" ++ show (kellyFraction mandraSignal) ++ " | PosLimit=" ++ show (positionLimit mandraSignal)
            threadDelay 50000

-- ========================================
-- LAUNCH MANDRA
-- ========================================
main :: IO ()
main = do
    let mandra = createMandraKernels 100000.0
    atomically $ riskGovernanceLoop mandra
