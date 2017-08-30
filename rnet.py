import preprocess
from Models import model_rnet
import numpy as np
import tensorflow as tf
import argparse
import random
import string
import os
import json

def run():
	parser = argparse.ArgumentParser()
	parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning Rate')
	parser.add_argument('--epochs', type=int, default=12, help='Expochs')
	parser.add_argument('--debug', type=bool, default=False, help='print debug msgs')
	parser.add_argument('--load', type=bool, default=False, help='load model')
	parser.add_argument('--save_dir', type=str, default='Models/save/', help='Data')

	args = parser.parse_args()

	modOpts = json.load(open('Models/config.json','r'))['rnet']['train']

	print('Reading data')
	dp = preprocess.read_data('train', modOpts)
	num_batches = int(np.floor(dp.num_samples/modOpts['batch_size'])) - 1
	
	rnet_model = model_rnet.R_NET(modOpts)
	input_tensors, loss, acc, pred_si, pred_ei = rnet_model.build_model()
	train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)

	#saver
	saver = tf.train.Saver()

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.InteractiveSession(config=config)
	if args.load:
		PATH = 'Models/save/rnet_model0.ckpt'
		start_epoch = 1
		saver.restore(sess, PATH)
		f = open('Results/rnet_training_result.txt','a')
	else:
		init = tf.global_variables_initializer()
		sess.run(init)
		f = open('Results/rnet_training_result.txt','w')
		start_epoch = 0

	for i in range(start_epoch, args.epochs):
		rl=random.sample(range(num_batches), num_batches)
		batch_no = 0
		LOSS = 0.0
		EM = 0.0
		while batch_no < num_batches:
			tensor_dict, idxs = dp.get_training_batch(rl[batch_no])
			_, loss_value, accuracy, predictions_si, predictions_ei = sess.run([train_op, loss, acc, pred_si, pred_ei], feed_dict={
				input_tensors['p']:tensor_dict['paragraph'],
				input_tensors['q']:tensor_dict['question'],
				input_tensors['pc']:tensor_dict['paragraph_c'],
				input_tensors['qc']:tensor_dict['question_c'],
				input_tensors['a_si']:tensor_dict['answer_si'],
				input_tensors['a_ei']:tensor_dict['answer_ei'],
			})
			batch_no += 1
			LOSS += loss_value
			EM += accuracy
			print("{} epoch {} batch, Loss:{:.2f}, Acc:{:.2f}".format(i, batch_no, loss_value, accuracy))
		save_path = saver.save(sess, os.path.join(args.save_dir, "rnet_model{}.ckpt".format(i)))
		f.write(' '.join( ("Loss", str(LOSS/dp.num_samples), str(i), '\n' ) ) )
		f.write(' '.join( ("EM", str(EM/num_batches), '\n') ) )
		f.write("---------------\n")
		f.flush()
		print("---------------")
	f.close()
	save_path = saver.save(sess, os.path.join(args.save_dir, "rnet_model_final.ckpt"))
	print('save path:',save_path)

def f1_score(prediction, ground_truth):
	from collections import Counter

	prediction_tokens = prediction
	ground_truth_tokens = ground_truth
	common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
	num_same = sum(common.values())
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(prediction_tokens)
	recall = 1.0 * num_same / len(ground_truth_tokens)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1

if __name__ == '__main__':
	run()
