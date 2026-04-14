// IB-specific: Requires SecurityID instead of Symbol
pub fn encode_ib(order: &NewOrderSingle) -> Vec<u8> {
    let mut msg = Vec::with_capacity(512);
    
    // IB uses FIX.4.2 with custom tags
    append_tag(&mut msg, 8, "FIX.4.2");
    append_tag(&mut msg, 9, "00000");
    append_tag(&mut msg, 35, "D");
    append_tag(&mut msg, 49, "RPDIB01");
    append_tag(&mut msg, 56, "IB");
    append_tag(&mut msg, 34, seq_num());
    append_tag(&mut msg, 52, utc_timestamp());
    append_tag(&mut msg, 11, &order.cl_ord_id);
    
    // IB-specific: SecurityID (tag 48) instead of Symbol (55)
    append_tag(&mut msg, 48, "885");  // Copper CFD ID
    append_tag(&mut msg, 22, "8");    // IDSource = EXCHANGE SYMBOL
    
    // Or use Symbol with proper format
    // append_tag(&mut msg, 55, "COPPER");
    
    append_tag(&mut msg, 54, side_char(order.side));
    append_tag(&mut msg, 38, &order.order_qty.to_string());
    append_tag(&mut msg, 40, ord_type_char(order.ord_type));
    append_tag(&mut msg, 59, tif_char(order.time_in_force));
    
    // IB requires Currency (15) and Exchange (207)
    append_tag(&mut msg, 15, "USD");
    append_tag(&mut msg, 207, "IDEALPRO");  // IB's metals venue
    
    append_tag(&mut msg, 60, &order.transact_time.to_string());
    append_checksum(&mut msg);
    
    msg
}
