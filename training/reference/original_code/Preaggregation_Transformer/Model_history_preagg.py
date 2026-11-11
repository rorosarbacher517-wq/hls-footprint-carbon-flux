import numpy as np
import tensorflow as tf
# import tensorflow_addons as tfa
# from keras import backend as K
# from keras.callbacks import TensorBoard
# from keras import backend as K
# from keras.callbacks import TensorBoard
import logging
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard


class ScaledSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Scale another schedule, used to match Keras AdamW decay semantics."""

    def __init__(self, schedule, scale):
        super().__init__()
        self.schedule = schedule
        self.scale = float(scale)

    def __call__(self, step):
        return tf.cast(self.scale, tf.float32) * self.schedule(step)

    def get_config(self):
        return {"scale": self.scale}


# logs_dir = "./fluxlogs" + 'logs'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=logs_dir, histogram_freq=1)

# 适应一个变量的准确度
class Maskedaccuracy(tf.keras.metrics.Metric):
    def __init__(self, mask_value=-9999, **kwargs):
        super(Maskedaccuracy, self).__init__(**kwargs)
        self.mask_value = mask_value
        self.total_mae = self.add_weight(name='total_mae', initializer='zeros')
        self.num_samples = self.add_weight(name='num_samples', initializer='zeros')
    def update_state(self, y_true, y_pred,sample_weight =None):
        y_true_flux = y_true[:,:,:3]
        is_mask = tf.equal(y_true_flux, self.mask_value)
        is_mask = tf.cast(is_mask, dtype=K.floatx())
        is_mask = 1 - is_mask
        y_true_masked = y_true_flux * is_mask
        y_pred_masked = y_pred      * is_mask
        masked_samples = tf.reduce_sum(is_mask, axis=tf.range(1, tf.rank(y_true_flux)))
        mae = tf.abs(y_true_masked - y_pred_masked)
        masked_mae = mae * is_mask
        sum_masked_mae = tf.reduce_sum(masked_mae, axis=tf.range(1, tf.rank(y_true_masked)))
        self.total_mae.assign_add(tf.reduce_sum(sum_masked_mae))
        self.num_samples.assign_add(tf.reduce_sum(masked_samples))
    def result(self):
        return self.total_mae / self.num_samples
    def reset_state(self):
        self.total_mae.assign(0.0)
        self.num_samples.assign(0.0)

# # 适应三个变量的
# class Maskedaccuracy(tf.keras.metrics.Metric):
#     def __init__(self, mask_value=-9999, **kwargs):
#         super(Maskedaccuracy, self).__init__(**kwargs)
#         self.mask_value = mask_value
#         self.total_accuracy = self.add_weight(name='total_accuracy', initializer='zeros')
#         self.num_samples = self.add_weight(name='num_samples', initializer='zeros')
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         is_mask = tf.equal(y_true, self.mask_value)
#         is_mask = tf.cast(is_mask, dtype=K.floatx())
#         is_mask = 1 - is_mask
#         y_true_masked = y_true * is_mask
#         y_pred_masked = y_pred * is_mask
#         masked_samples = tf.reduce_sum(is_mask, axis=list(range(1, K.ndim(y_true))))
#         accuracy_var = K.equal(tf.round(y_true_masked), tf.round(y_pred_masked))
#         accuracy_var = tf.cast(accuracy_var, dtype=K.floatx())
#         accuracy_var = accuracy_var * is_mask
#         sum_masked_accuracy = tf.reduce_sum(accuracy_var, axis=list(range(1, K.ndim(y_true_masked))))
#         self.total_accuracy.assign_add(tf.reduce_sum(sum_masked_accuracy))
#         self.num_samples.assign_add(tf.reduce_sum(masked_samples))
#
#     def result(self):
#         return self.total_accuracy / self.num_samples
#
#     def reset_states(self):
#         self.total_accuracy.assign(0.0)
#         self.num_samples.assign(0.0)


## https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/
def lr_warmup_cosine_decay(global_step, warmup_steps, hold=0, total_steps=0, target_lr=1e-3, start_lr=0.0):
    # Cosine decay
    # There is no tf.pi so we wrap np.pi as a TF constant
    global_step_tf = tf.cast(global_step, tf.float32)
    learning_rate = 0.5 * target_lr * (1 + tf.cos(tf.constant(np.pi) * (global_step_tf - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = tf.cast(target_lr * (global_step / warmup_steps), tf.float32)

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = tf.where(global_step > warmup_steps + hold, learning_rate, target_lr)

    learning_rate = tf.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return tf.cast(learning_rate, tf.float32)


# https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/
class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    #首先使用余弦函数（tf.cos()）实现余弦衰减，然后使用tf.cast()函数将学习率转换为tf.float32类型，
    # 最后，使用tf.where()函数根据global_step的值来选择是否继续使用热启动学习率或余弦衰减学习率
    def __init__(self, start_lr, target_lr, warmup_steps, total_steps, hold):
        super().__init__()
        self.start_lr = start_lr # 初始学习率
        self.target_lr = target_lr # 目标学习率
        self.warmup_steps = warmup_steps # 学习率热启动的步数，即在这些步数内学习率会逐渐增加
        self.total_steps = total_steps # 总步数，代表训练的总迭代数
        self.hold = hold # 学习率在达到目标学习率后保持恒定的步数

    def __call__(self, step):
        # 根据输入的step值，通过调用lr_warmup_cosine_decay函数计算学习率lr。这个函数实现了余弦衰减和学习率热启动的逻辑。
        lr = lr_warmup_cosine_decay(global_step=step, total_steps=self.total_steps, warmup_steps=self.warmup_steps,
                                    start_lr=self.start_lr,
                                    target_lr=self.target_lr, hold=self.hold)
        # # 最后，通过使用tf.where函数来判断是否超过了总步数，超过则返回学习率为0.0，否则返回计算得到的学习率
        return tf.where(step > self.total_steps, 0.0, lr, name="learning_rate")
# @tf.keras.utils.register_keras_serializable(package="Custom")
# class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, start_lr, target_lr, warmup_steps, total_steps, hold):
#         super().__init__()
#         self.start_lr = start_lr
#         self.target_lr = target_lr
#         self.warmup_steps = warmup_steps
#         self.total_steps = total_steps
#         self.hold = hold
#
#     def __call__(self, step):
#         lr = lr_warmup_cosine_decay(
#             global_step=step,
#             total_steps=self.total_steps,
#             warmup_steps=self.warmup_steps,
#             start_lr=self.start_lr,
#             target_lr=self.target_lr,
#             hold=self.hold,
#         )
#         return tf.where(step > self.total_steps, 0.0, lr, name="learning_rate")
#
#     def get_config(self):
#         # Enable Keras to save the parameters of this schedule into the configuration when using `model.save()`.
#         return {
#             "start_lr": self.start_lr,
#             "target_lr": self.target_lr,
#             "warmup_steps": self.warmup_steps,
#             "total_steps": self.total_steps,
#             "hold": self.hold,
#         }
#
#     @classmethod
#     def from_config(cls, config):
#         # Keras calls this during deserialization (it often works even without specifying it, but specifying it ensures the most stable operation).
#         return cls(**config)

# 定义了一个名为LRTensorBoard的自定义TensorBoard回调类，它继承自TensorBoard类。
# # 在每个训练周期结束时记录学习率的值到TensorBoard日志文件中，以便可视化学习率的变化。
class LRTensorBoard(TensorBoard):
    # 在构造函数__init__中，它接收一个log_dir参数和其他可选参数 **kwargs，
    # 并通过调用父类的构造函数super().__init__(log_dir=log_dir, **kwargs)来初始化TensorBoard回调。
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)
    # # 在on_epoch_end方法中，回调函数会在每个训练周期结束时调用。
    def on_epoch_end(self, epoch, logs=None):
        # 在这个方法中，它首先初始化logs为一个空字典（如果logs参数没有提供的话）
        logs = logs or {}
        # 获取当前的优化器optimizer和学习率current_lr
        optimizer = self.model.optimizer
        # 检查学习率是否是tf.keras.optimizers.schedules.LearningRateSchedule类型的实例
        # if isinstance(optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
        if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            # 调用学习率调度器，传入当前迭代步数，得到当前学习率张量
            current_lr = optimizer.learning_rate(optimizer.iterations)
        else:
            # 直接取固定学习率
            current_lr = optimizer.learning_rate

        # 评估当前学习率的具体数值（从张量转成 numpy/scalar）
        current_lr_val = tf.keras.backend.get_value(current_lr)

        logs.update({'lr': current_lr_val})

        logging.info(logs)
        super().on_epoch_end(epoch, logs)

def my_train_1schedule(model, HLS_train_norm1,FLUX_train_norm1, epochs=10, start_rate=0.001, loss=np.nan, per_epoch=100,
                       split_epoch=4, option=0, decay=1e-5,
                       batch_size=32, validation_split=0.04, hold_epoch=0, reduce_epoch=False):
    momentum = 0.8# 动量
    momentum = 0.9# 动量
    # 创建了一个字符串print_str，用于打印和记录有关动量、学习率和衰减的信息
    print_str = "momentum=" + str(momentum) + "\tlearning rate=" + str(start_rate) + "  decay=" + str(decay)
    # 调用print函数和logging.info函数将print_str打印和记录到日志中。
    print(print_str);
    logging.info(print_str)
    ##*****************************************************************************************************
    ## warm up
    # warm_up = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=start_rate/split_epoch/per_epoch,decay_steps=split_epoch*per_epoch,end_learning_rate=start_rate,name='Decay_linear')
    total_steps = per_epoch * epochs # 表示总步数，即每个周期内的总迭代次数乘以总周期数
    warmup_steps = per_epoch * split_epoch # 表示热启动的步数,在这些步数内学习率会逐渐增加
    hold_steps = per_epoch * hold_epoch # hold_steps表示学习率保持恒定的步数
    # 学习率衰减策略对象schedule_one 控制训练过程中学习率的变化。
    schedule_one = WarmUpCosineDecay(start_lr=0.0, target_lr=start_rate, warmup_steps=warmup_steps,
                                     total_steps=total_steps, hold=hold_steps)
    if option == 0:
        print('tfa.optimizers.RMSprop');
        # 实例化一个tf.keras.optimizers.RMSprop对象，将衰减率decay、学习率schedule_one和动量rho设置为给定的数值。
        # optimizer = tf.keras.optimizers.RMSprop(decay=decay,learning_rate=schedule_one, rho=momentum);
        optimizer = tf.keras.optimizers.legacy.RMSprop(decay=decay,learning_rate=schedule_one, rho=momentum)
    elif option == 1:
        # 实例化一个tf.keras.optimizers.Adam对象，将学习率schedule_one、一阶矩估计的动量beta_1、二阶矩估计的动量beta_2和epsilon设置为给定的数值。
        print('tfa.optimizers.Adam ');
        optimizer = tf.keras.optimizers.Adam(learning_rate=schedule_one, beta_1=momentum, beta_2=0.999, epsilon=1e-07)
    else:
        # 实例化一个tfa.optimizers.AdamW对象，将权重衰减因子weight_decay、学习率schedule_one和动量beta_1设置为给定的数值。
        print('tfa.optimizers.AdamW ');
        if hasattr(tf.optimizers, "AdamW"):
            optimizer = tf.optimizers.AdamW(
                weight_decay=decay, learning_rate=schedule_one, beta_1=momentum
            )
        else:
            # TensorFlow 2.8 (the local GPU environment) does not expose
            # tf.optimizers.AdamW. TensorFlow Addons provides the same
            # decoupled-weight-decay optimizer used by the original design.
            import tensorflow_addons as tfa
            optimizer = tfa.optimizers.AdamW(
                # TFA subtracts weight_decay * variable directly, whereas
                # Keras AdamW uses learning_rate * weight_decay * variable.
                weight_decay=ScaledSchedule(schedule_one, decay),
                learning_rate=schedule_one,
                beta_1=momentum,
            )
    
    # 设置了模型的编译参数，包括优化器（optimizer）、损失函数（loss）和评估指标（metrics）。使用model.compile函数将这些参数应用于模型。
    model.compile(optimizer = optimizer, loss=loss, metrics=[Maskedaccuracy()])
    # model_history = model.fit(train_dataset, validation_data=validation_dataset, epochs=split_epoch, verbose=2)
    if reduce_epoch:
        # 如果reduce_epoch为True，则将最终的训练周期数减去10，但不会小于10。
        final_epochs = (epochs - 10) if epochs > 10 else epochs
    else:
        # 否则，最终的训练周期数等于初始设定的周期数。
        final_epochs = epochs
    # 然后，根据是否设置了验证集的比例，使用适当的参数调用model.fit方法来训练模型。同时，使用LRTensorBoard回调函数来记录训练过程中的学习率信息。
    # 将输入数据转为tensor
    # 将numpy数组转换为tf.Tensor并创建tf.data.Dataset对象
    train_datasetx = [HLS_train_norm1, FLUX_train_norm1]

    print(type(FLUX_train_norm1))  # 应该是 tf.Tensor 或 np.ndarray
    print(np.array(FLUX_train_norm1).ndim)  # 维度数
    print(np.array(FLUX_train_norm1).dtype)  # 数据类型
    train_datasety = tf.gather(FLUX_train_norm1, indices=[7, 8, 9, 80], axis=-1)
    if validation_split > 0:
        # 训练集数据(train_datasetx和train_datasety)，验证集，最终的训练周期数(final_epochs)、输出详细信息的级别(verbose)、批量大小(batch_size)以及一个回调函数LRTensorBoard来记录训练过程中的学习率和其他信息。
        model_history = model.fit(x=train_datasetx, y = train_datasety, validation_split=validation_split, epochs=final_epochs,
                                  verbose=2, batch_size=batch_size, callbacks=[LRTensorBoard(log_dir="./tmp/tb_log")])
        # model_history = model.fit(x=train_datasetx, y=train_datasety, validation_split=0.1, epochs=final_epochs,
        #                           verbose=2, batch_size=batch_size, callbacks=[tensorboard_callback])
    else:
        model_history = model.fit(x=train_datasetx, y = train_datasety, epochs=final_epochs, verbose=2,
                                  batch_size=batch_size, callbacks=[LRTensorBoard(log_dir="./tmp/tb_log")])
        # model_history = model.fit(x=train_datasetx, y=train_datasety, epochs=final_epochs, verbose=2,
        #                           batch_size=batch_size, callbacks=[tensorboard_callback])
    # print(model_history )
    return model_history
