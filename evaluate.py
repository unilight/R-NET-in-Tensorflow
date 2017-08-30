import preprocess
from Models import model_rnet
import numpy as np
import tensorflow as tf
import argparse
import random
import json
from pprint import pprint

def run():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='rnet', help='Model: match_lstm, bidaf, rnet')
	parser.add_argument('--debug', type=bool, default=False, help='print debug msgs')
	parser.add_argument('--dataset', type=str, default='dev', help='dataset')
	parser.add_argument('--model_path', type=str, default='Models/save/rnet_model0.ckpt', help='saved model path')

	args = parser.parse_args()
	if not args.model == 'rnet':
		raise NotImplementedError

	modOpts = json.load(open('Models/config.json','r'))[args.model]['dev']
	print('Model Configs:')
	pprint(modOpts)

	print('Reading data')
	if args.dataset == 'train':
		raise NotImplementedError
	elif args.dataset == 'dev':
		dp = preprocess.read_data(args.dataset, modOpts)
    
	model = model_rnet.R_NET(modOpts)
	input_tensors, loss, acc, pred_si, pred_ei = model.build_model()
	saved_model = args.model_path


	num_batches = int(np.ceil(dp.num_samples/modOpts['batch_size']))
	print(num_batches, 'batches')
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	new_saver = tf.train.Saver()
	sess = tf.InteractiveSession(config=config)
	new_saver.restore(sess, saved_model)
	
	pred_data = {}

	EM = 0.0
	F1 = 0.0
	empty_answer_idx = np.ndarray((modOpts['batch_size'], modOpts['p_length']))
	for batch_no in range(num_batches):
		if args.model == 'rnet':
			context, context_original, paragraph, question, paragraph_c, question_c, answer_si, answer_ei, ID, n = dp.get_testing_batch(batch_no)
			predictions_si, predictions_ei = sess.run([pred_si, pred_ei], feed_dict={
				input_tensors['p']:paragraph,
				input_tensors['q']:question,
				input_tensors['pc']:paragraph_c,
				input_tensors['qc']:question_c,
				input_tensors['a_si']:empty_answer_idx,
				input_tensors['a_ei']:empty_answer_idx,
			})
		for i in range(n):
			parag = context[i]
			f1 = []
			p_tokens = []
			for j in range(len(answer_si[i])):
				if answer_si[i][j] == answer_ei[i][j]: # single word answer
					truth_tokens = [parag[int(answer_si[i][j])]]
					pred_tokens = [parag[int(predictions_si[i])]]
				else:
					truth_tokens = parag[int(answer_si[i][j]):int(answer_ei[i][j])+1]
					pred_tokens = parag[int(predictions_si[i]):int(predictions_ei[i])+1]
				f1.append(f1_score( pred_tokens, truth_tokens ))
				p_tokens.append(pred_tokens)
			idx = np.argmax(f1)
			if answer_si[i][idx] == int(predictions_si[i]) and answer_ei[i][idx] == int(predictions_ei[i]):
				EM += 1.0
			F1 += f1[idx]
			pred_data[ID[i]] =  ' '.join( p_tokens[idx] )
		print(batch_no, 'EM', '{:.5f}'.format(EM/(batch_no+1)/modOpts['batch_size']), 'F1', F1/(batch_no+1)/modOpts['batch_size'])
	print("---------------")
	print("EM", EM/dp.num_samples )
	print("F1", F1/dp.num_samples )
	with open('Results/'+args.model+'_prediction.txt', 'w') as outfile:
	    json.dump(pred_data, outfile)
    
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
