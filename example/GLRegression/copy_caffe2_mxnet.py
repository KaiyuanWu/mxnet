#[in_height x in_width x in_channel x out_channel] --> [out_channel x in_channel x in_height x in_width]
def copy_conv(src, dst):
  assert len(src.shape) == len(dst.shape) == 4
  dst[:] = np.rollaxis(np.rollaxis(src,3,0), 3, 1)

#previous layer is a conv layer
#[in_channel x out_channel] --> [out_channel x in_channel]
def copy_fc_1(src, dst):
  dst[:] = np.rollaxis(np.rollaxis(src,1,0), 3,2)
  
def copy_fc_1(src, dst):
  dst[:] = np.rollaxis(src,1,0)
  
