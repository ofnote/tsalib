
def modeling_embedding_lookup(input_ids: 'bti'):
    # illustrates local dim var usage, i is not declared globaly
    B, T, D = dim_vars('B(b):13 L(t):7 D(d):32')
    embedding_size = D
    i = get_shape_list(input_ids)[-1] #num inputs
    #TODO: define/pickup i from input_ids

    output: 'b*t*i,d'

    # OLD
    input_shape: 'bti' = get_shape_list(input_ids)
    output: 'btd' = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * embedding_size])

    #NEW
    #no need for input_shape
    output: 'btd' = warp(output, tfms=f'b*t*{i},d -> b,t,d*{i}', tfm_names='r')
  
    #assert output.get_shape() == (B, T, D)


def create_attention_mask_from_input_mask_old(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask


def create_attention_mask_from_input_mask(from_tensor: 'b,f,...', to_mask: 'b,t'):

    B, F = get_shape_list(from_tensor, expected_rank=[2, 3])[:2]
    To = get_shape_list(to_mask, expected_rank=2)[1]
    to_mask = alignto((to_mask, 'bt'), 'b_t')
    to_mask = tf.cast(to_mask, tf.float32)
    broadcast_ones = tf.ones(shape=[B, F, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask
    size_assert (get_shape_list(mask), (B, F, To))
    print (f'create attn: {get_shape_list(mask)}')

    return mask
