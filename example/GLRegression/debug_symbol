def debug_symbol(sym):
    '''Get internals values for blobs (forward only).'''
    args = sym.list_arguments()
    output_names  = [] 
    
    sym = sym.get_internals()
    blob_names = sym.list_outputs()
    sym_group = []
    for i in range(len(blob_names)):
        if blob_names[i] not in args:
            x = sym[i]
            if blob_names[i] not in output_names:
                x = mx.symbol.BlockGrad(x, name=blob_names[i])
            sym_group.append(x)
    sym = mx.symbol.Group(sym_group)
    return sym
