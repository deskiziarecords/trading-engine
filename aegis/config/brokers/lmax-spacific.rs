// LMAX-specific tag ordering (cache-optimized)
pub fn encode_lmax(order: &NewOrderSingle) -> Vec<u8> {
    // LMAX requires specific sequence for fast path
    let mut msg = Vec::with_capacity(256);
    
    // Required sequence: 8,9,35,49,56,34,52,11,55,54,38,40,59,60,10
    append_tag(&mut msg, 8, "FIX.4.4");
    append_tag(&mut msg, 9, "00000"); // Body length placeholder
    append_tag(&mut msg, 35, "D");     // NewOrderSingle
    append_tag(&mut msg, 49, "RPD_TRADER_01");
    append_tag(&mut msg, 56, "LMAX");
    append_tag(&mut msg, 34, seq_num());
    append_tag(&mut msg, 52, utc_timestamp());
    append_tag(&mut msg, 11, &order.cl_ord_id);  // ClOrdID
    append_tag(&mut msg, 55, "XCUUSD");          // Symbol
    append_tag(&mut msg, 54, side_char(order.side));
    append_tag(&mut msg, 38, &order.order_qty.to_string());
    append_tag(&mut msg, 40, ord_type_char(order.ord_type));
    append_tag(&mut msg, 59, tif_char(order.time_in_force));
    append_tag(&mut msg, 60, &order.transact_time.to_string());
    
    // LMAX-specific: Account tag (required)
    append_tag(&mut msg, 1, "RPD_ACCT_001");
    
    // Checksum
    append_checksum(&mut msg);
    
    msg
}
