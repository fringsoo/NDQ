import torch as th
import torch.nn as nn
import torch.nn.functional as F
from seq2seq.models import DecoderRNN
from seq2seq.models import EncoderRNN
import numpy as np

class IPComm_mind(nn.Module):
	def __init__(self, input_shape, args):
		super(IPComm_mind, self).__init__()
		self.args = args
		self.n_agents = args.n_agents
		
		self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
		self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
		self.fc3 = nn.Linear(args.rnn_hidden_dim, args.comm_embed_dim)
		
		self.inference_model = nn.Sequential(
			nn.Linear(input_shape + args.comm_embed_dim * self.n_agents, 4 * args.comm_embed_dim * self.n_agents),
			nn.ReLU(True),
			nn.Linear(4 * args.comm_embed_dim * self.n_agents, 4 * args.comm_embed_dim * self.n_agents),
			nn.ReLU(True),
			nn.Linear(4 * args.comm_embed_dim * self.n_agents, args.n_actions)
		)

		self.fc = nn.Linear(input_shape, args.comm_embed_dim)

	def forward(self, inputs):
		x = F.relu(self.fc1(inputs))
		x = F.relu(self.fc2(x))
		gaussian_params = self.fc3(x)

		#gaussian_params = self.fc(inputs)
		mu = gaussian_params
		return mu


class IPComm_speaker_train(nn.Module):
	def __init__(self, input_shape, args):
		super(IPComm_speaker_train, self).__init__()	
		self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
		self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
		self.fc3 = nn.Linear(args.rnn_hidden_dim, args.comm_embed_dim)

	def forward(self, inputs):
		x = F.relu(self.fc1(inputs))
		x = F.relu(self.fc2(x))
		gaussian_params = self.fc3(x)

		#gaussian_params = self.fc(inputs)
		mu = gaussian_params
		return mu


class IPComm_listener_train(nn.Module):
	def __init__(self, args):
		super(IPComm_listener_train, self).__init__()	
		#self.fc1 = nn.Linear(args.comm_embed_dim, args.comm_embed_dim)
	
	def forward(self, message):
		#x = self.fc1(message)
		return message

class IPComm_speaker(nn.Module):
	def __init__(self, args):
		super(IPComm_speaker, self).__init__()
		self.vocab_size = 10
		self.max_len = 5
		self.hidden_size = args.comm_embed_dim
		self.eos_id = 1
		self.sos_id = 0
		self.speaker = DecoderRNN(self.vocab_size, self.max_len, self.hidden_size, eos_id=self.eos_id, sos_id=self.sos_id, rnn_cell='gru')

	def forward(self, mu):
		decode = self.speaker(None, mu.clone().view(1, mu.shape[0], -1))
		#message = th.Tensor(np.array([bt.cpu().numpy() for bt in decode[2]['sequence']]).transpose(1,0,2).squeeze()).long().cuda()
		message = np.array([bt.cpu().numpy() for bt in decode[2]['sequence']]).transpose(1,0,2).squeeze()

		return message

class IPComm_listener(nn.Module):
	def __init__(self, args):
		super(IPComm_listener, self).__init__()
		self.vocab_size = 10
		self.max_len = 5
		self.hidden_size = args.comm_embed_dim
		self.listener = EncoderRNN(self.vocab_size, self.max_len, self.hidden_size, rnn_cell='gru')
		self.reconstruct_fc = nn.Linear(self.hidden_size, args.comm_embed_dim)

		self.fc = nn.Linear(args.comm_embed_dim, args.comm_embed_dim)

	def forward(self, message):
		return self.fc(message)
		#message_e = self.listener(message)[1].squeeze().view(message.shape[0], -1)
		#mu_reconstruct = F.relu(self.reconstruct_fc(message_e))
		#return mu_reconstruct

class IPComm_speaker_simple(nn.Module):
	def __init__(self, args):
		super(IPComm_speaker_simple, self).__init__()	
		#self.fc1 = nn.Linear(args.comm_embed_dim, 4 * args.comm_embed_dim)
		#self.fc2 = nn.Linear(4 * args.comm_embed_dim, args.comm_embed_dim)
		self.fc = nn.Linear(args.comm_embed_dim, args.comm_embed_dim)


	def forward(self, mu):
		#x = F.relu(self.fc1(mu))
		#x = self.fc2(x)
		#import pdb; pdb.set_trace()
		x = self.fc(mu)
		return x

class IPComm_listener_simple(nn.Module):
	def __init__(self, args):
		super(IPComm_listener_simple, self).__init__()	
		self.fc1 = nn.Linear(args.comm_embed_dim, args.comm_embed_dim)
	
	def forward(self, message):
		x = self.fc1(message)
		return x