# myloss
import numpy as np
import tensorflow as tf
# from keras import backend as K
from tensorflow.keras import backend as K

# y_pred=model2.predict(X_rnn_train[:16,:,:])
# y_true=y_rnn_train[:16,:,:]
# maskValue=-999
def mask_loss(maskValue=-9999.0):
    def mask_loss_in(y_true, y_pred):
        isMask = K.equal(y_true, maskValue)  # true for all mask values
        isMask = K.cast(isMask, dtype=K.floatx())
        isMask = 1 - isMask  # now mask values are zero, and others are 1
        y_true = y_true * isMask
        y_pred = y_pred * isMask
        axis_to_reduce = range(1, K.ndim(y_true))
        sum_square = K.sum(K.square(y_true - y_pred), axis=axis_to_reduce)
        sum_n = K.sum(isMask, axis=axis_to_reduce)
        loss1 = K.sqrt(sum_square / sum_n)
        return loss1
    
    return mask_loss_in


# maskValue=-9999
# inputs = (input_images_train_norm3 [:2,:,:,:,:], input_covariate_train_norm3[:2,:])
# y_pred = model.predict(inputs)
# y_true = tf.convert_to_tensor(trainy[:2,:,:].astype(np.float))
# y_true = trainy[:2,:,:].astype(np.float)
# RMSE
# def mask_loss_physical(maskValue=-9999.0):
#     lambda1 = 0.1
#     lambda1 = 1
#     # lambda1 = 10
#     def mask_loss_in(y_true, y_pred):
#         # 首先通过K.equal(y_true, maskValue),# 判断y_true中是否存在填充的值，返回一个布尔类型的张量。
#         isMask = K.equal(y_true, maskValue)  # true for all mask values
#         # 使用K.cast将其转换为与K.floatx()数据类型相同的张量，取反，然后乘以mask返回y_true,这样填充值就被mask
#         isMask = K.cast(isMask, dtype=K.floatx())
#         isMask = 1 - isMask  # now mask values are zero, and others are 1
#         y_true = y_true * isMask
#         # loss1根据公式计算了预测值y_pred与真实值y_true之间的均方根误差。
#         # loss #1
#         y_pred1 = y_pred * isMask
#         axis_to_reduce = range(1, K.ndim(y_true))
#         sum_square1 = K.sum(K.square(y_true - y_pred1), axis=axis_to_reduce)
#         sum_n1 = K.sum(isMask, axis=axis_to_reduce)
#         loss1 = K.sqrt(sum_square1 / sum_n1)
#         # # loss2根据物理模型限制（NEE + GPP -RECO）计算了一个新的损失，然后得到均方根误差
#         # loss #2
#         isMask2 = tf.cast(tf.math.greater(K.sum(isMask, axis=2), 0), tf.float32)
#         sum_square2 = K.sum(K.square(y_pred[:, :, 0] + y_pred[:, :, 1] - y_pred[:, :, 2]) * isMask2, axis=1)
#         sum_n2 = K.sum(isMask2, axis=1)
#         loss2 = K.sqrt(sum_square2 / sum_n2)
#         return loss1 + lambda1 * loss2
#     return mask_loss_in

# maskValue=-9999
# inputs = [HLS_test_tensor[:2,:,:,:], flux_test_numeric_tensor[:2,:,:] ]
# y_pred = model.predict(inputs)
# cols = [7, 8, 9, 53]
# y_true = tf.gather(flux_test_numeric_tensor[:2], indices=cols, axis=-1)# grab those feature columns along the last axis
# lambda1=1.0; beta=1.0

def mask_loss_physical_gpp(maskValue=-9999.0, lambda1=1.0, beta=1.0):
    # lambda1 = 1  # you can adjust this
    SSRD_mean=245.659
    SSRD_std =309.24582
    def mask_loss_in(y_true, y_pred):
        # create mask: 1 for valid data, 0 for mask value
        y_true_flux = y_true[:,:,:3]
        isMask = tf.not_equal(y_true_flux, maskValue)  # boolean mask
        isMask = tf.cast(isMask, dtype=tf.keras.backend.floatx())  # float mask: 1/0
        
        y_true_masked = y_true_flux * isMask
        y_pred_masked = y_pred      * isMask
        
        ##*****************************************************
        # loss #1: RMSE for valid values
        axis_to_reduce = tf.range(1, tf.rank(y_true_flux))  # e.g., reduce over [1, 2] for shape (batch, time, vars)
        sum_square1 = tf.reduce_sum(tf.square(y_true_masked - y_pred_masked), axis=axis_to_reduce)
        sum_n1 = tf.reduce_sum(isMask, axis=axis_to_reduce)
        loss1 = tf.sqrt(sum_square1 / (sum_n1 + 1e-6))  # add epsilon to avoid NaN
        
        ##*****************************************************
        # loss #2: physical constraint (NEE + GPP - RECO) per timestep
        # y_pred: (batch, time, 3)
        # constraint = NEE + GPP - RECO ≈ 0
        constraint = y_pred[:, :, 0] + y_pred[:, :, 1] - y_pred[:, :, 2]
        # valid_timestep_mask = tf.reduce_sum(isMask, axis=2) > 0  # shape: (batch, time)
        valid_timestep_mask = tf.reduce_sum(isMask, axis=2) > -1  # shape: (batch, time) changed to apply all predictions
        valid_timestep_mask = tf.cast(valid_timestep_mask, tf.float32)
        sum_n2 = tf.reduce_sum(valid_timestep_mask, axis=1)
        sum_square2 = tf.reduce_sum(tf.square(constraint) * valid_timestep_mask, axis=1)
        loss2 = tf.sqrt(sum_square2 / (sum_n2 + 1e-6))
        
        ##*****************************************************
        # loss #3: GPP = 0 when ssrd is less than 2 
        ssrd = y_true[:,:,3]*SSRD_std+SSRD_mean
        mask = tf.cast(ssrd < 3, y_pred.dtype)
        loss3 = tf.reduce_mean(mask * tf.square(y_pred[:,:,1]),axis=1)
        return loss1 + lambda1 * loss2 + beta * loss3 
        
    return mask_loss_in



def mask_loss_physical(maskValue=-9999.0,lambda1=1.0):
    # lambda1 = 1  # you can adjust this
    
    def mask_loss_in(y_true, y_pred):
        # create mask: 1 for valid data, 0 for mask value
        isMask = tf.not_equal(y_true, maskValue)  # boolean mask
        isMask = tf.cast(isMask, dtype=tf.keras.backend.floatx())  # float mask: 1/0
        
        y_true_masked = y_true * isMask
        y_pred_masked = y_pred * isMask
        
        # loss #1: RMSE for valid values
        axis_to_reduce = tf.range(1, tf.rank(y_true))  # e.g., reduce over [1, 2] for shape (batch, time, vars)
        sum_square1 = tf.reduce_sum(tf.square(y_true_masked - y_pred_masked), axis=axis_to_reduce)
        sum_n1 = tf.reduce_sum(isMask, axis=axis_to_reduce)
        loss1 = tf.sqrt(sum_square1 / (sum_n1 + 1e-6))  # add epsilon to avoid NaN
        
        # loss #2: physical constraint (NEE + GPP - RECO) per timestep
        # y_pred: (batch, time, 3)
        # constraint = NEE + GPP - RECO ≈ 0
        constraint = y_pred[:, :, 0] + y_pred[:, :, 1] - y_pred[:, :, 2]
        # valid_timestep_mask = tf.reduce_sum(isMask, axis=2) > 0  # shape: (batch, time)
        valid_timestep_mask = tf.reduce_sum(isMask, axis=2) > -1  # shape: (batch, time) changed to apply all predictions
        valid_timestep_mask = tf.cast(valid_timestep_mask, tf.float32)
        sum_n2 = tf.reduce_sum(valid_timestep_mask, axis=1)
        sum_square2 = tf.reduce_sum(tf.square(constraint) * valid_timestep_mask, axis=1)
        loss2 = tf.sqrt(sum_square2 / (sum_n2 + 1e-6))
        
        return loss1 + lambda1 * loss2
        
    return mask_loss_in


def mask_loss_physical_one_variable(maskValue=-9999.0,lambda1=1.0):
    # lambda1 = 1  # you can adjust this

    def mask_loss_in(y_true, y_pred):
        # create mask: 1 for valid data, 0 for mask value
        isMask = tf.not_equal(y_true, maskValue)  # boolean mask
        isMask = tf.cast(isMask, dtype=tf.keras.backend.floatx())  # float mask: 1/0
        
        y_true_masked = y_true * isMask
        y_pred_masked = y_pred * isMask
        
        # loss #1: RMSE for valid values
        axis_to_reduce = tf.range(1, tf.rank(y_true))  # e.g., reduce over [1, 2] for shape (batch, time, vars)
        sum_square1 = tf.reduce_sum(tf.square(y_true_masked[:,:,1:2] - y_pred_masked[:,:,1:2]), axis=axis_to_reduce)
        sum_n1 = tf.reduce_sum(isMask, axis=axis_to_reduce)
        loss1 = tf.sqrt(sum_square1 / (sum_n1 + 1e-6))  # add epsilon to avoid NaN
        
        # loss #2: physical constraint (NEE + GPP - RECO) per timestep
        # y_pred: (batch, time, 3)
        # constraint = NEE + GPP - RECO ≈ 0
        constraint = y_pred[:, :, 0] + y_pred[:, :, 1] - y_pred[:, :, 2]
        valid_timestep_mask = tf.reduce_sum(isMask, axis=2) > 0  # shape: (batch, time)
        valid_timestep_mask = tf.cast(valid_timestep_mask, tf.float32)
        sum_square2 = tf.reduce_sum(tf.square(constraint) * valid_timestep_mask, axis=1)
        sum_n2 = tf.reduce_sum(valid_timestep_mask, axis=1)
        loss2 = tf.sqrt(sum_square2 / (sum_n2 + 1e-6))
        
        return loss1 + lambda1 * loss2
        
    return mask_loss_in


# maskValue=-9999
# inputs = (input_images_train_norm3 [:2,:,:,:,:], input_covariate_train_norm3[:2,:])
# y_pred = model.predict(inputs)
# y_true = tf.convert_to_tensor(trainy[:2,:,:].astype(np.float32))
# y_true = trainy[:2,:,:].astype(np.float32)
def mask_loss_physical_MAE(maskValue=-9999.0,lambda1 = 1):
    # lambda1 = 0.1
    # lambda1 = 1
    
    # lambda1 = 10
    def mask_loss_in(y_true, y_pred):
        isMask = K.equal(y_true, maskValue)  # true for all mask values
        isMask = K.cast(isMask, dtype=K.floatx())
        isMask = 1 - isMask  # now mask values are zero, and others are 1
        y_true = y_true * isMask
        # loss #1
        y_pred1 = y_pred * isMask
        axis_to_reduce = range(1, K.ndim(y_true))
        sum_square1 = K.sum(K.abs(y_true - y_pred1), axis=axis_to_reduce)
        sum_n1 = K.sum(isMask, axis=axis_to_reduce)
        loss1 = sum_square1 / (sum_n1 + 1e-6)
        # loss #2
        isMask2 = tf.cast(tf.math.greater(K.sum(isMask, axis=2), 0), tf.float32)
        # sum_square2 = K.sum(K.abs(y_pred[:, :, 0] + y_pred[:, :, 1] - y_pred[:, :, 2]) * isMask2, axis=1)
        sum_square2 = K.sum(K.abs(y_true[:, :, 0] + y_true[:, :, 1] - y_true[:, :, 2]) * isMask2, axis=1)
        # sum_square2 = K.sum(K.abs(y_pred[:, :, 2] - y_pred[:, :, 1] - y_pred[:, :, 0]) * isMask2, axis=1)
        sum_n2 = K.sum(isMask2, axis=1)
        loss2 = sum_square2 / (sum_n2 + 1e-6)
        return loss1 + lambda1 * loss2
    
    return mask_loss_in

def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = tf.keras.backend.abs(error) < clip_delta
  squared_loss = 0.5 * tf.keras.backend.square(error)
  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
  return tf.where(cond, squared_loss, linear_loss)

# def mask_loss_huber(maskValue=-9999.0):
#     lambda1 = 0.1
#     lambda1 = 1
#     # lambda1 = 10
#     def mask_loss_in(y_true, y_pred):
#         # 首先通过K.equal(y_true, maskValue),# 判断y_true中是否存在填充的值，返回一个布尔类型的张量。
#         isMask = K.equal(y_true, maskValue)  # true for all mask values
#         # 使用K.cast将其转换为与K.floatx()数据类型相同的张量，取反，然后乘以mask返回y_true,这样填充值就被mask
#         isMask = K.cast(isMask, dtype=K.floatx())
#         isMask = 1 - isMask  # now mask values are zero, and others are 1
#         y_true = y_true * isMask
#         # loss1根据公式计算了预测值y_pred与真实值y_true之间的均方根误差。
#         # loss #1
#         y_pred1 = y_pred * isMask
#         axis_to_reduce = range(0, K.ndim(y_true))
#         losses = []
#         for i in axis_to_reduce:
#             # 修改为Huber Loss的计算方式
#             # Huber loss替换loss1
#             loss = huber_loss(y_true[:, :, i], y_pred1[:, :, i],clip_delta=1.0)  # Calculate Huber Loss for each variable
#             loss = K.mean(loss, axis=1)
#             losses.append(loss)
#         # Convert the list of tensors to a single tensor with shape (3, 1684)
#         losses_tensor = tf.convert_to_tensor(losses)
#         # Transpose the tensor to shape (1684, 3)
#         losses_transposed = tf.transpose(losses_tensor)
#         # Calculate the mean across axis 1, resulting in a tensor with shape (1684, )
#         loss1 = tf.reduce_mean(losses_transposed, axis=1)
#         # # loss2根据物理模型限制（NEE + GPP -RECO）计算了一个新的损失，然后得到均方根误差
#         # loss #2
#         isMask2 = tf.cast(tf.math.greater(K.sum(isMask, axis=2), 0), tf.float32)
#         sum_square2 = K.sum(K.square(y_pred[:, :, 0] + y_pred[:, :, 1] - y_pred[:, :, 2]) * isMask2, axis=1)
#         sum_n2 = K.sum(isMask2, axis=1)
#         loss2 = K.sqrt(sum_square2 / sum_n2)
#         return loss1 + lambda1 * loss2
#     return mask_loss_in

def mask_loss_huber(maskValue=-9999.0):
    lambda1 = 0.1
    lambda1 = 1
    # lambda1 = 10
    def mask_loss_in(y_true, y_pred):
        # 首先通过K.equal(y_true, maskValue),# 判断y_true中是否存在填充的值，返回一个布尔类型的张量。
        isMask = K.equal(y_true, maskValue)  # true for all mask values
        # 使用K.cast将其转换为与K.floatx()数据类型相同的张量，取反，然后乘以mask返回y_true,这样填充值就被mask
        isMask = K.cast(isMask, dtype=K.floatx())
        isMask = 1 - isMask  # now mask values are zero, and others are 1
        y_true = y_true * isMask
        # loss1根据公式计算了预测值y_pred与真实值y_true之间的均方根误差。
        # loss #1
        y_pred1 = y_pred * isMask
        axis_to_reduce = range(0, K.ndim(y_true))
        losses = []
        for i in axis_to_reduce:
            # 修改为Huber Loss的计算方式
            loss = huber_loss(y_true[:, :, i], y_pred1[:, :, i],
                              clip_delta=1.0)  # Calculate Huber Loss for each variable
            loss = K.mean(loss, axis=1)
            losses.append(loss)
        # Convert the list of tensors to a single tensor with shape (3, 1684)
        losses_tensor = tf.convert_to_tensor(losses)
        # Transpose the tensor to shape (1684, 3)
        losses_transposed = tf.transpose(losses_tensor)
        # Calculate the mean across axis 1, resulting in a tensor with shape (1684, )
        loss1 = tf.reduce_mean(losses_transposed, axis=1)
        # # loss2根据物理模型限制（NEE + GPP -RECO）计算了一个新的损失
        # loss #2
        isMask2 = tf.cast(tf.math.greater(K.sum(isMask, axis=2), 0), tf.float32)
        y_pred2 = (y_pred[:, :, 0] + y_pred[:, :, 1]) * isMask2
        y_pred3 = y_pred[:, :, 2] * isMask2
        # Huber loss替换loss2
        loss2 = huber_loss(y_pred2, y_pred3, clip_delta=1.0)  # Calculate Huber Loss
        loss2 = tf.reduce_mean(loss2, axis=1)
        return loss1 + lambda1 * loss2
    return mask_loss_in