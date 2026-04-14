import jax
import jax.numpy as jnp
from aegis.zeta_flux.akts import AdelicKoopmanSynchronizer, QSH42Config, AdelicConfig, MTPConfig, simulate_market_signals

def main():
    print("=" * 80)
    print("AEGIS: Autonomous Execution & Generalized Intelligence System")
    print("=" * 80)

    # Initialize configurations
    qsh_config = QSH42Config()
    adelic_config = AdelicConfig()
    mtp_config = MTPConfig()

    print(f"\n[CONFIG] QSH-42 Architecture initialized.")
    print(f"[CONFIG] Adelic Space & MTP Foresight configured.")

    # Initialize synchronizer
    synchronizer = AdelicKoopmanSynchronizer(qsh_config, adelic_config, mtp_config)

    # Generate test data
    print("\n[DATA] Generating synthetic market signals...")
    key = jax.random.PRNGKey(42)
    signals, mtp = simulate_market_signals(key=key)

    # Run inference (JIT compile and execute)
    print("[EXEC] Running forward pass...")

    @jax.jit
    def run_inference(s, m):
        return synchronizer(s, m, return_all=True)

    results = run_inference(signals, mtp)

    print("\n[RESULTS] Trajectory Synchronization:")
    print(f"  - Output Shape: {results['trajectories'].shape}")
    print(f"  - Tube Integrity: {float(results['tube_integrity']):.4f}")

    print("\n" + "=" * 80)
    print("AEGIS system ready for deployment")
    print("=" * 80)

if __name__ == "__main__":
    main()
