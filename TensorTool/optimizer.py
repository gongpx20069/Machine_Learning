import tensorflow as tf

def Adam_optimizer(loss,variables,starter_learning_rate = 0.0002,end_learning_rate = 0.0,start_decay_step = 100000,decay_steps = 100000,beta1 = 0.5,name='Adam',summary=True):
	""" Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
		and a linearly decaying rate that goes to zero over the next 100k steps
	"""
	'''
	loss: 计算的损失函数的值
	varivables：需要修改的模型变量名字（通常使用tf.get_collection来收集成列表）
	starter_learning_rate：刚开始训练时候的学习率
	end_learning_rate：最后的学习率
	start_decay_step：开始衰减学习率的步数
	decay_steps：在多少步以内线性衰减学习率
	beta1：Adam优化器中的beta1
	name：优化器名字
	summary：是否将学习率大小summary，便于在tensorboard中查看
	'''
	global_step = tf.Variable(0, trainable=False)
	learning_rate = (
		tf.where(
			tf.greater_equal(global_step, start_decay_step),
			tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
									decay_steps, end_learning_rate,
									power=1.0),
			starter_learning_rate
		)
	)
	if summary:
		tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)
	learning_step = (
	tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
						.minimize(loss, global_step=global_step, var_list=variables)
	)
	return learning_step

def BGD_optimizer(loss,variables,starter_learning_rate = 0.0002,end_learning_rate = 0.0,start_decay_step = 100000,decay_steps = 100000,name='Batch-GD',summary=True):
	'''
	loss: 计算的损失函数的值
	varivables：需要修改的模型变量名字（通常使用tf.get_collection来收集成列表）
	starter_learning_rate：刚开始训练时候的学习率
	end_learning_rate：最后的学习率
	start_decay_step：开始衰减学习率的步数
	decay_steps：在多少步以内线性衰减学习率
	name：优化器名字
	summary：是否将学习率大小summary，便于在tensorboard中查看
	'''
	global_step = tf.Variable(0, trainable=False)
	learning_rate = (
		tf.where(
			tf.greater_equal(global_step, start_decay_step),
			tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
									decay_steps, end_learning_rate,
									power=1.0),
			starter_learning_rate
		)
	)
	if summary:
		tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)
	learning_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate,name=name).minimize(loss,var_list=variables,global_step=global_step)
	return learning_step
