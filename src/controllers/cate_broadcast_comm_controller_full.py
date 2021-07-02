from modules.agents import REGISTRY as agent_REGISTRY
from modules.comm import REGISTRY as comm_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.distributions as D
import torch.nn.functional as F
import time
import numpy as np
import random


# This multi-agent controller shares parameters between agents
class CateBCommFMAC:
	def __init__(self, scheme, groups, args):
		self.n_agents = args.n_agents
		self.args = args
		self.comm_embed_dim = args.comm_embed_dim
		input_shape_for_comm = 0
		if args.comm:
			input_shape, input_shape_for_comm = self._get_input_shape(scheme)
		else:
			input_shape = self._get_input_shape(scheme)
		print(input_shape)

		self._build_agents(input_shape)
		self.agent_output_type = args.agent_output_type

		self.action_selector = action_REGISTRY[args.action_selector](args)

		self.hidden_states = None

		if args.comm:
			self.comm = comm_REGISTRY[self.args.comm_method](input_shape_for_comm, args)
			comm_prag = comm_REGISTRY['information_pragmatics']
			self.comm_mind = comm_prag[0](input_shape_for_comm, args)
			self.mu_t0 = th.zeros([args.n_agents, args.comm_embed_dim])
			# self.comm_speakers = []
			# self.comm_listeners = []
			# for na in range(self.n_agents):
			# 	self.comm_speakers.append(comm_prag[1](args))
			# 	self.comm_listeners.append(comm_prag[2](args))
			
			# self.comm_speakers_simple = []
			# self.comm_listeners_simple = []
			# for na in range(self.n_agents):
			# 	self.comm_speakers_simple.append(comm_prag[3](args))
			# 	self.comm_listeners_simple.append(comm_prag[4](args))

			# self.comm_speakers_train = []
			# self.comm_listeners_train = []
			# for na in range(self.n_agents):
			# 	self.comm_speakers_train.append(comm_prag[5](input_shape_for_comm, args))
			# 	self.comm_listeners_train.append(comm_prag[6](args))

		self.s_mu = th.zeros(1)
		self.s_sigma = th.ones(1)

		self.is_print_once = False

		if self.args.env_args['print_rew']:
			self.c_step = 0
			self.cut_mean_list = []
			self.cut_mean_num_list = []

	def _print(self, data, l_thres, r_thres):
		data = data.detach().cpu().squeeze().numpy()
		for i in range(self.n_agents):
			data[i, i * self.args.comm_embed_dim: (i + 1) * self.args.comm_embed_dim] = 0
		index = np.argsort(abs(data).reshape(-1))
		rank = np.zeros_like(index)
		for i, item in enumerate(index):
			rank[item] = i
		rank = 100. * rank / rank.size
		rank = rank.reshape(data.shape)
		data = np.round(data, decimals=2)
		rank = np.round(rank)
		rank[abs(data) >= l_thres] = 0
		rank[abs(data) <= r_thres] = 0
		data[abs(data) >= l_thres] = 0
		data[abs(data) <= r_thres] = 0
		for i in range(self.n_agents):
			for j in range(self.n_agents):
				if i != j:
					print('agent %d to agent %d:' % (i + 1, j + 1),
					      data[i, j * self.args.comm_embed_dim: (j + 1) * self.args.comm_embed_dim],
					      rank[i, j * self.args.comm_embed_dim: (j + 1) * self.args.comm_embed_dim])

	def select_actions_old(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, thres=0., prob=0.):
		# Only select actions for the selected batch elements in bs
		avail_actions = ep_batch["avail_actions"][:, t_ep]
		if self.args.comm:
			agent_outputs, (mu, sigma), _, m_sample = self.test_forward(ep_batch, t_ep, test_mode=test_mode,
			                                                            thres=thres, prob=prob)
			if self.args.env_args['print_rew']:
				data = mu.detach().cpu().squeeze().numpy()
				s = 0
				s_num = 0
				#r_thres = self.args.cut_mu_thres
				r_thres = thres
				for i in range(self.n_agents):
					for j in range(self.n_agents):
						if i != j:
							flag = 1
							for k in range(self.args.comm_embed_dim):
								xxx = abs(data[i, j * self.args.comm_embed_dim + k])
								if xxx < r_thres or random.random() < prob:
									s += 1
								else:
									flag = 0
							s_num += flag
				self.c_step += 1
				self.cut_mean_list.append(1. * s / (self.n_agents * (self.n_agents - 1) * self.args.comm_embed_dim
				                                    + 1e-3))
				self.cut_mean_num_list.append(1. * s_num / (self.n_agents * (self.n_agents - 1) + 1e-3))
				if self.c_step >= self.args.env_args['print_steps'] and not self.is_print_once:
					print('cut ratio:', np.mean(self.cut_mean_list))
					print('cut num ratio:', np.mean(self.cut_mean_num_list))
					self.c_step = 0
					self.is_print_once = True
			if self.args.is_print:
				print('------------------------')
				print('mu')
				self._print(mu, l_thres=100., r_thres=self.args.cut_mu_thres)
				# print('sigma')
				# self.print(sigma, l_thres=0.1, r_thres=0.)
				# print('KL')
				# KL = D.kl_divergence(D.Normal(mu, sigma), D.Normal(self.s_mu, self.s_sigma))
				# self.print(KL, l_thres=1e4, r_thres=2.)
				time.sleep(1)
		else:
			agent_outputs = self.test_forward(ep_batch, t_ep, test_mode=test_mode, thres=thres, prob=prob)
		chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
		                                                    test_mode=test_mode)
		return chosen_actions

	def test_forward_old(self, ep_batch, t, test_mode=False, thres=0., prob=0.):
		agent_inputs = self._build_inputs(ep_batch, t)

		mu, sigma, logits, m_sample = None, None, None, None

		if self.args.comm:
			(mu, sigma), messages, m_sample = self._test_communicate(ep_batch.batch_size, agent_inputs,
			                                                         thres=thres, prob=prob)
			agent_inputs = th.cat([agent_inputs, messages], dim=1)
			logits = self._logits(ep_batch.batch_size, agent_inputs)

		avail_actions = ep_batch["avail_actions"][:, t]
		agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

		# shape = (bs, self.n_agents, -1)
		if self.args.comm:
			agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
			mu = mu.view(ep_batch.batch_size, self.n_agents, -1)
			sigma = sigma.view(ep_batch.batch_size, self.n_agents, -1)
			logits = logits.view(ep_batch.batch_size, self.n_agents, -1)
			m_sample = m_sample.view(ep_batch.batch_size, self.n_agents, -1)
			return agent_outs, (mu, sigma), logits, m_sample
		else:
			return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

	def forward_old(self, ep_batch, t, test_mode=False):
		agent_inputs = self._build_inputs(ep_batch, t)

		mu, sigma, logits, m_sample = None, None, None, None

		if self.args.comm:
			(mu, sigma), messages, m_sample = self._communicate(ep_batch.batch_size, agent_inputs)
			agent_inputs = th.cat([agent_inputs, messages], dim=1)
			logits = self._logits(ep_batch.batch_size, agent_inputs)

		avail_actions = ep_batch["avail_actions"][:, t]
		agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

		# Softmax the agent outputs if they're policy logits
		# is not used in qmix
		if self.agent_output_type == "pi_logits":

			if getattr(self.args, "mask_before_softmax", True):
				# Make the logits for unavailable actions very negative to minimise their affect on the softmax
				reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
				agent_outs[reshaped_avail_actions == 0] = -1e10

			agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
			if not test_mode:
				# Epsilon floor
				epsilon_action_num = agent_outs.size(-1)
				if getattr(self.args, "mask_before_softmax", True):
					# With probability epsilon, we will pick an available action uniformly
					epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

				agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
				              + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)

				if getattr(self.args, "mask_before_softmax", True):
					# Zero out the unavailable actions
					agent_outs[reshaped_avail_actions == 0] = 0.0

		# shape = (bs, self.n_agents, -1)
		if self.args.comm:
			agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
			mu = mu.view(ep_batch.batch_size, self.n_agents, -1)
			sigma = sigma.view(ep_batch.batch_size, self.n_agents, -1)
			logits = logits.view(ep_batch.batch_size, self.n_agents, -1)
			m_sample = m_sample.view(ep_batch.batch_size, self.n_agents, -1)
			return agent_outs, (mu, sigma), logits, m_sample
		else:
			return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

	def forward(self, ep_batch, t, test_mode=False, cut_mu=False, prag=False, thres=0., prob=0.):
		agent_inputs = self._build_inputs(ep_batch, t)

		if True:
			mu = self.comm_mind(agent_inputs)
			
			if not test_mode and t==0:
				mu_t0 = mu.view([ep_batch.batch_size, self.n_agents, self.comm_embed_dim]).sum(axis=0) / ep_batch.batch_size
				self.mu_t0 = 0.9 * self.mu_t0 + 0.1 * mu_t0
			

			mu_inputs = mu.repeat(1, self.n_agents)
			sigma = th.ones(mu_inputs.shape).cuda()
			

			if prag==False:
				normal_distribution = D.Normal(mu_inputs, sigma)
				ms = normal_distribution.rsample()
				#ms = mu_inputs
				if cut_mu:
					ms = self._cut_mu_new(ms, prob=prob)
					#ms = self._cut_mu(mu_inputs, ms, thres=thres, prob=prob)

			else:
				if t == 0:
					#self.mu_prev = th.zeros_like(mu)
					self.mu_prev = self.mu_t0.repeat(ep_batch.batch_size,1,1).view(ep_batch.batch_size * self.n_agents, self.comm_embed_dim)
				ms = mu - self.mu_prev
				if cut_mu:
					ms = self._cut_mu_new(ms, prob=prob)
					#ms = self._cut_mu(ms, ms, thres=thres, prob=prob)
				ms = self.mu_prev + ms
				self.mu_prev = ms
				ms = ms.repeat(1, self.n_agents)
			

			message = ms.clone().view(ep_batch.batch_size, self.n_agents, self.n_agents, -1).permute(0, 2, 1, 3).contiguous().view(ep_batch.batch_size * self.n_agents, -1)
			agent_inputs = th.cat([agent_inputs, message], dim=1)
		
			#t_logits = self.comm.inference_model(agent_inputs)
			t_logits = self.comm_mind.inference_model(agent_inputs)
			logits = F.softmax(t_logits, dim=1).view(ep_batch.batch_size, self.n_agents, -1)
			sigma = sigma.view(ep_batch.batch_size, self.n_agents, -1)
			mu_inputs = mu_inputs.view(ep_batch.batch_size, self.n_agents, -1)
			ms = ms.view(ep_batch.batch_size, self.n_agents, -1)
			info = {'mu':mu_inputs, 'sigma':sigma, 'logits':logits, 'm_sample':ms}


		'''
		elif False:
			mu = self.comm_mind(agent_inputs) #(bs*na, ed)
			
			mu_in_agents =  mu.view(ep_batch.batch_size, self.n_agents, -1).permute(1, 0, 2) #(na, bs, ed)
			
			messages = np.array([self.comm_speakers[na](mu_in_agents[na]) for na in range(self.n_agents)]) #(na, bs, md)
			messages = messages.transpose(1, 0 ,2) #(bs, na, md)
			messages = messages.repeat(1, 1, self.n_agents) #(bs, na, nar*md)
			messages = messages.reshape(ep_batch.batch_size, self.n_agents, self.n_agents, -1) #(ba, na, nar, md)
			messages = messages.transpose(2, 0, 1, 3) #(nar, ba, na, md)

			messages_use = [self.comm_listeners[na](messages[na]) for na in range(self.n_agents)] #(nar, ba, na, ed)
			messages_use = messages_use.permute(1, 0, 2, 3)

			message_use = mu_inputs.clone().view(ep_batch.batch_size, self.n_agents, self.n_agents, -1).permute(0, 2, 1, 3).contiguous().view(ep_batch.batch_size * self.n_agents, -1)
			
			messages_reconstruct = [messages[:, na, na, :] for na in range(self.n_agents)]
			agent_inputs = th.cat([agent_inputs, messages_use], dim=1)
		
			reconstruct_loss = messages_reconstruct - mu_in_agents

			#mu_prev = None
			#mu = self.comm_mind(agent_inputs)
			#short_train(comm_speaker, comm_listener, mu, mu_prev)

			sigma = th.ones(mu_inputs.shape).cuda()
			t_logits = self.comm_mind.inference_model(agent_inputs)
			logits = F.softmax(t_logits, dim=1).view(ep_batch.batch_size, self.n_agents, -1)
			sigma = sigma.view(ep_batch.batch_size, self.n_agents, -1)
			mu_inputs = mu_inputs.view(ep_batch.batch_size, self.n_agents, -1)
			ms = ms.view(ep_batch.batch_size, self.n_agents, -1)
			info = {'mu':mu_inputs, 'sigma':sigma, 'logits':logits, 'm_sample':ms}

		elif False:
			mu_inputs = agent_inputs.view(ep_batch.batch_size, self.n_agents, -1)			
			mu_inputs = th.stack([self.comm_speakers_train[na](mu_inputs[:, na, :]) for na in range(self.n_agents)]) #(na, bs, md)
			messages = mu_inputs
			

			if cut_mu:
				messages = self._cut_mu(messages, messages, thres=thres, prob=prob)


			messages_use = th.stack([self.comm_listeners_train[na](messages) for na in range(self.n_agents)]) #(nar, na, bs, ed)
			messages_use = messages_use.permute(1, 2, 0, 3).contiguous().view(ep_batch.batch_size * self.n_agents, -1)
			agent_inputs = th.cat([agent_inputs, messages_use], dim=1)
			

			ms = messages.permute(1, 0, 2).contiguous().view(ep_batch.batch_size, self.n_agents, -1).repeat(1,1,self.n_agents)
			sigma = th.ones(ms.shape).cuda()
			t_logits = self.comm_mind.inference_model(agent_inputs)
			logits = F.softmax(t_logits, dim=1).view(ep_batch.batch_size, self.n_agents, -1)

			info = {'mu':ms, 'sigma':sigma, 'logits':logits, 'm_sample':ms}
		
		elif False:
			mu = self.comm_mind(agent_inputs)  #(bs*na, ed)

			#speaker
			mu_inputs = mu.view(ep_batch.batch_size, self.n_agents, -1) #(bs, na, ed)
			mu_inputs = th.stack([self.comm_speakers_simple[na](mu_inputs[:, na, :]) for na in range(self.n_agents)]) #(na, bs, md)
			mu_inputs = mu_inputs.permute(1, 0 ,2)
			#speaker

			messages = mu_inputs.cpu().detach().numpy()
			messages = th.Tensor(messages).cuda()
			
			messages = th.stack([self.comm_listeners_simple[na](messages) for na in range(self.n_agents)]) #(nar, ba, na, ed)
			
			messages_use = messages.permute(1, 0, 2, 3).contiguous().view(ep_batch.batch_size * self.n_agents, -1)
			#messages_use = mu.repeat(1, self.n_agents).view(ep_batch.batch_size, self.n_agents, self.n_agents, -1).permute(0,2,1,3).contiguous().view(ep_batch.batch_size * self.n_agents, -1)
			#messages_use = th.rand(ep_batch.batch_size*self.n_agents, self.n_agents*self.args.comm_embed_dim).cuda()

			agent_inputs = th.cat([agent_inputs, messages_use], dim=1)
		
			messages_reconstruct = th.stack([messages[na, :, na, :].squeeze() for na in range(self.n_agents)])
			messages_reconstruct = th.stack([self.comm_listeners_simple[na](mu_inputs[:, na, :].squeeze())  for na in range(self.n_agents)])
			messages_reconstruct = messages_reconstruct.permute(1, 0, 2).contiguous().view(ep_batch.batch_size * self.n_agents, -1)
			reconstruct_loss = messages_reconstruct - mu
			
			# mu_inputs = None #sf(mu)
			# s_habit_loss = None #mu_inputs
			# cost_loss = None #mu_inputs
			# message_use = None #lf(mu_inputs)
			# message_reconstruct = None #lf(mu_inputs)
			# l_habit_loss = None #message_use
			# reconstruct_loss = None #mu, message_reconstruct
			# reconstruct_game_loss = None #mu, messsage_use
			# speaker_loss = s_habit_loss + cost_loss
			# listener_loss = l_habit_loss + acc_loss
			# realuse_loss = None #message_use
			

			mu_inputs = mu.repeat(1, self.n_agents)  #(bs*na, nar*ed)
			ms = messages.permute(0,2,1,3).contiguous().view(ep_batch.batch_size * self.n_agents, -1)  #(bs*na, nar*ed)
			#ms = messages.view(ep_batch.batch_size, self.n_agents, self.n_agents, -1).permute(0,2,1,3).contiguous().view(ep_batch.batch_size * self.n_agents, -1)  #(bs*na, nar*ed)
			sigma = th.ones(ms.shape).cuda()
			t_logits = self.comm_mind.inference_model(agent_inputs)
			
			logits = F.softmax(t_logits, dim=1).view(ep_batch.batch_size, self.n_agents, -1)
			sigma = sigma.view(ep_batch.batch_size, self.n_agents, -1)
			ms = ms.contiguous().view(ep_batch.batch_size, self.n_agents, -1)
			mu_inputs = mu_inputs.view(ep_batch.batch_size, self.n_agents, -1)
			info = {'mu':ms, 'sigma':sigma, 'logits':logits, 'm_sample':ms, 'reconstruct_loss':reconstruct_loss}
		
		else:
			info = {}
		'''
		
		avail_actions = ep_batch["avail_actions"][:, t]
		agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
		return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), info
		#return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), (mu_inputs, sigma), logits, mu_inputs


	def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, prag=False, thres=0., prob=0.):
		# Only select actions for the selected batch elements in bs
		avail_actions = ep_batch["avail_actions"][:, t_ep]
		agent_outputs, info = self.forward(ep_batch, t_ep, test_mode=test_mode, cut_mu=True, prag=prag, thres=thres, prob=prob)
		chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
		return chosen_actions

	def init_hidden(self, batch_size):
		self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

	def parameters(self):
		return list(self.agent.parameters()) + list(self.comm_mind.parameters())
		#return list(self.agent.parameters()) + list(self.comm_mind.parameters()) + self.parameters_speakerlistener_train()
		#return self.parameters_speakerlistener_simple()

	def parameters_speakerlistener_simple(self):
		l = list()
		for na in range(self.n_agents):
			l += list(self.comm_speakers_simple[na].parameters())
			l += list(self.comm_listeners_simple[na].parameters())
		return l

	def parameters_speakerlistener_train(self):
		l = list()
		for na in range(self.n_agents):
			l += list(self.comm_speakers_train[na].parameters())
			l += list(self.comm_listeners_train[na].parameters())
		return l
		
	def load_state(self, other_mac):
		self.agent.load_state_dict(other_mac.agent.state_dict())
		self.comm.load_state_dict(other_mac.comm.state_dict())

	def cuda(self):
		self.agent.cuda()
		self.comm.cuda()
		self.comm_mind.cuda()
		self.mu_t0 = self.mu_t0.cuda()
		# for cs in self.comm_speakers: cs.cuda()
		# for cl in self.comm_listeners: cl.cuda()
		# for cs in self.comm_speakers_simple: cs.cuda()
		# for cl in self.comm_listeners_simple: cl.cuda()
		# for cs in self.comm_speakers_train: cs.cuda()
		# for cl in self.comm_listeners_train: cl.cuda()
		self.s_mu = self.s_mu.cuda()
		self.s_sigma = self.s_sigma.cuda()

	def save_models(self, path):
		th.save(self.agent.state_dict(), "{}/agent.th".format(path))
		th.save(self.comm.state_dict(), "{}/comm.th".format(path))
		th.save(self.comm_mind.state_dict(), "{}/comm_mind.th".format(path))
		th.save(self.mu_t0, "{}/mu_t0.th".format(path))

		# for na in range(self.n_agents):
		# 	th.save(self.comm_speakers[na].state_dict(), "{}/comm_speaker_{}.th".format(path, na))
		# 	th.save(self.comm_listeners[na].state_dict(), "{}/comm_listener_{}.th".format(path, na))
		# for na in range(self.n_agents):
		# 	th.save(self.comm_speakers_simple[na].state_dict(), "{}/comm_speaker_simple_{}.th".format(path, na))
		# 	th.save(self.comm_listeners_simple[na].state_dict(), "{}/comm_listener_simple_{}.th".format(path, na))
		# for na in range(self.n_agents):
		# 	th.save(self.comm_speakers_train[na].state_dict(), "{}/comm_speaker_train_{}.th".format(path, na))
		# 	th.save(self.comm_listeners_train[na].state_dict(), "{}/comm_listener_train_{}.th".format(path, na))
			
	def load_models(self, path):
		self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
		self.comm.load_state_dict(th.load("{}/comm.th".format(path), map_location=lambda storage, loc: storage))
		self.comm_mind.load_state_dict(th.load("{}/comm_mind.th".format(path), map_location=lambda storage, loc: storage))
		self.mu_t0 = th.load("{}/mu_t0.th".format(path))

		#for na in range(self.n_agents):
		#	self.comm_speakers[na].load_state_dict(th.load("{}/comm_speaker_{}.th".format(path, na), map_location=lambda storage, loc: storage))
		#	self.comm_listeners[na].load_state_dict(th.load("{}/comm_listener_{}.th".format(path, na), map_location=lambda storage, loc: storage))
		#for na in range(self.n_agents):
		#	self.comm_speakers_simple[na].load_state_dict(th.load("{}/comm_speaker_simple_{}.th".format(path, na), map_location=lambda storage, loc: storage))
		#	self.comm_listeners_simple[na].load_state_dict(th.load("{}/comm_listener_simple_{}.th".format(path, na), map_location=lambda storage, loc: storage))
		#for na in range(self.n_agents):
		#	self.comm_speakers_train[na].load_state_dict(th.load("{}/comm_speaker_train_{}.th".format(path, na), map_location=lambda storage, loc: storage))
		#	self.comm_listeners_train[na].load_state_dict(th.load("{}/comm_listener_train_{}.th".format(path, na), map_location=lambda storage, loc: storage))

	def _build_agents(self, input_shape):
		self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

	def _build_inputs(self, batch, t):
		# Assumes homogenous agents with flat observations.
		# Other MACs might want to e.g. delegate building inputs to each agent
		# shape = (bs * self.n_agents, -1)
		bs = batch.batch_size
		inputs = []
		inputs.append(batch["obs"][:, t])  # b1av
		if self.args.obs_last_action:
			if t == 0:
				inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
			else:
				inputs.append(batch["actions_onehot"][:, t - 1])
		if self.args.obs_agent_id:
			inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

		inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
		return inputs

	def _get_input_shape(self, scheme):
		# shape = (bs * self.n_agents, -1 + self.comm_embed_dim * self.n_agents)
		input_shape = scheme["obs"]["vshape"]
		ms_shape = 0
		if self.args.obs_last_action:
			input_shape += scheme["actions_onehot"]["vshape"][0]
		if self.args.obs_agent_id:
			input_shape += self.n_agents
		if self.args.comm:
			ms_shape = self.comm_embed_dim * self.n_agents

		if self.args.comm:
			return input_shape + ms_shape, input_shape
		else:
			return input_shape

	def _cut_mu(self, mu, ms, thres=0., prob=0.):
		mu = mu.detach().cpu()
		ms = ms.detach().cpu()
		if self.args.is_rank_cut_mu:
			index = np.argsort(abs(mu).reshape(-1))
			rank = np.zeros_like(index)
			for i, item in enumerate(index):
				rank[item] = i
			rank = 100. * rank / rank.size
			rank = th.Tensor(rank.reshape(mu.shape))
			# ms[rank <= self.args.cut_mu_rank_thres] = 0.
			ms[rank < thres] = 0.
		else:
			#r_thres = self.args.cut_mu_thres
			r_thres = thres
			mu = mu.numpy()
			for i in range(self.n_agents):
				for j in range(self.n_agents):
					if i != j:
						for k in range(self.args.comm_embed_dim):
							index = j * self.args.comm_embed_dim + k
							if abs(mu[i, index]) < r_thres or random.random() < prob:
								ms[i, index] = 0.

		ms = ms.cuda()
		return ms

	def _cut_mu_new(self, ms, prob=0.):
		ms = ms.detach().cpu()
		
		f = np.random.random(ms.shape)>prob
		ms = ms * f

		ms = ms.cuda()
		return ms

	def _test_communicate(self, bs, inputs, thres=0., prob=0.):
		# shape = (bs * self.n_agents, -1)
		mu, sigma = self.comm(inputs)
		normal_distribution = D.Normal(mu, sigma)
		ms = normal_distribution.rsample()
		if self.args.is_cur_mu:
			ms = self._cut_mu(mu, ms, thres=thres, prob=prob)
		message = ms.clone().view(bs, self.n_agents, self.n_agents, -1)
		message = message.permute(0, 2, 1, 3).contiguous().view(bs * self.n_agents, -1)
		return (mu, sigma), message, ms

	def _communicate(self, bs, inputs):
		# shape = (bs * self.n_agents, -1)
		mu, sigma = self.comm(inputs)
		normal_distribution = D.Normal(mu, sigma)
		ms = normal_distribution.rsample()
		message = ms.clone().view(bs, self.n_agents, self.n_agents, -1)
		message = message.permute(0, 2, 1, 3).contiguous().view(bs * self.n_agents, -1)
		return (mu, sigma), message, ms

	def _logits(self, bs, inputs):
		# shape = (bs * self.n_agents, -1)
		t_logits = self.comm.inference_model(inputs)
		logits = F.softmax(t_logits, dim=1)
		return logits

	def clean(self):
		if self.args.env_args['print_rew']:
			self.c_step = 0
			self.cut_mean_list = []
			self.cut_mean_num_list = []
			self.is_print_once = False
