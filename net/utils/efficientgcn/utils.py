import math

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def rescale_block(block_args, scale_args, scale_factor):
    channel_scaler = math.pow(scale_args[0], scale_factor)
    depth_scaler = math.pow(scale_args[1], scale_factor)
    new_block_args = []
    for [channel, stride, depth] in block_args:
        channel = max(int(round(channel * channel_scaler / 16)) * 16, 16)
        depth = int(round(depth * depth_scaler))
        new_block_args.append([channel, stride, depth])
    return new_block_args