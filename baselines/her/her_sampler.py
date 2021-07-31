import numpy as np
import gym

def make_random_sample(reward_fun):
    def _random_sample(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout
        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                            for key in episode_batch.keys()}

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # # Re-compute reward since we may have substituted the u and o_2 ag_2
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                        for k in transitions.keys()}
        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions
    return _random_sample
        
def obs_to_goal_fun(env):
    # only support Fetchenv and Handenv now
    from gym.envs.robotics import FetchEnv, hand_env

    if isinstance(env.env, FetchEnv):
        obs_dim = env.observation_space['observation'].shape[0]
        goal_dim = env.observation_space['desired_goal'].shape[0]
        temp_dim = env.sim.data.get_site_xpos('robot0:grip').shape[0]
        def obs_to_goal(observation):
            observation = observation.reshape(-1, obs_dim)
            if env.has_object:
                goal = observation[:, temp_dim:temp_dim + goal_dim]
            else:
                goal = observation[:, :goal_dim]
            return goal.copy()
    elif isinstance(env.env, hand_env.HandEnv):
        goal_dim = env.observation_space['desired_goal'].shape[0]
        def obs_to_goal(observation):
            goal = observation[:, -goal_dim:]
            return goal.copy()
    else:
        raise NotImplementedError('Do not support such type {}'.format(env))
        
    return obs_to_goal


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, obs_to_goal_fun=None, no_her=False):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    if no_her:
        print( '*' * 10 + 'Do not use HER in this method' + '*' * 10)
    
    def _random_log(string):
        if np.random.random() < 0.002:
            print(string)
    
    def _preprocess(episode_batch, batch_size_in_transitions, std=None, use_std=False):
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout
        # Select which episodes and time steps to use. 
        if use_std:
            print(np.where(std > 1e-3)[0].shape)
            prob = std / (std.sum() + 1e-4)
            prob = prob / prob.sum()
            episode_idxs = np.random.choice(np.arange(rollout_batch_size), batch_size, p=prob)
            np.random.randint(0, rollout_batch_size, batch_size)
        else:
            episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}
        return transitions, episode_idxs, t_samples, batch_size, T

    def _get_reward(ag_2, g):
        # Reconstruct info dictionary for reward  computation.
        info = {}
        reward_params = {'ag_2':ag_2, 'g':g}
        reward_params['info'] = info
        return reward_fun(**reward_params)

    def _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T, future_p=future_p):
        her_indexes = (np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        return future_ag.copy(), her_indexes.copy()

    def _reshape_transitions(transitions, batch_size, batch_size_in_transitions):
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)
        return transitions


    def _sample_her_transitions(episode_batch, batch_size_in_transitions, info=None):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)
        
        if not no_her:
            future_ag, her_indexes = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T)
            transitions['g'][her_indexes] = future_ag

        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g'])
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)

    def _dynamic_interaction_full(o, g, action_fun, dynamic_model, steps):
        batch_size = o.shape[0]
        last_state = o.copy()
        states_list,actions_list, next_states_list = [], [], []
        goals_list, ags_list, next_ags_list, reward_list = [], [], [], []
        for _ in range(0, steps):
            goals_list.append(g.copy())
            states_list.append(last_state.copy())
            ag_array = obs_to_goal_fun(last_state).copy()
            ags_list.append(ag_array)

            action_array = action_fun(o=last_state, g=g) 
            action_array += 0.2 * np.random.randn(*action_array.shape)  # gaussian noise
            action_array = np.clip(action_array, -1, 1)
            next_state_array = dynamic_model.predict_next_state(last_state, action_array)

            actions_list.append(action_array.copy())
            next_states_list.append(next_state_array.copy())
            next_ag_array = obs_to_goal_fun(next_state_array).copy()
            next_ags_list.append(next_ag_array)
            reward_list.append(_get_reward(next_ag_array, g))
            last_state = next_state_array
        transitions = {}
        transitions['o'] = np.concatenate(states_list,axis=0).reshape(batch_size * steps, -1) 
        transitions['o_2'] = np.concatenate(next_states_list,axis=0).reshape(batch_size * steps, -1) 
        transitions['ag'] = np.concatenate(ags_list,axis=0).reshape(batch_size * steps, -1) 
        transitions['ag_2'] = np.concatenate(next_ags_list,axis=0).reshape(batch_size * steps, -1) 
        transitions['g'] = np.concatenate(goals_list,axis=0).reshape(batch_size * steps, -1) 
        transitions['r'] = np.concatenate(reward_list,axis=0).reshape(batch_size * steps, -1)
        transitions['u'] = np.concatenate(actions_list,axis=0).reshape(batch_size * steps, -1)
        return transitions

    def _sample_mbpo_transitions(episode_batch, batch_size_in_transitions, info):
        dynamic_model, action_fun, steps = info['dynamic_model'], info['action_fun'], info['nstep']
        model_samples_buffer = info['model_buffer']
        _random_log('using goal mbpo sampler with step:{}'.format(steps))

        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)
        if not no_her:
            future_ag, her_indexes = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T)
            transitions['g'][her_indexes] = future_ag
        model_transitions = _dynamic_interaction_full(transitions['o'], transitions['g'], action_fun, dynamic_model, steps)
        model_samples_buffer.store_transitions(model_transitions)
        sample_model_batches = model_samples_buffer.sample(batch_size)
        return _reshape_transitions(sample_model_batches, batch_size, batch_size_in_transitions)


    def _sample_hero_transitions(episode_batch, batch_size_in_transitions, info):
        steps, gamma, Q_fun, alpha, std, use_std = info['nstep'], info['gamma'], info['get_Q_pi'], info['alpha'], info['std'], info['use_std']
        dynamic_model, action_fun = info['dynamic_model'], info['action_fun']
        transitions, episode_idxs, t_samples, batch_size, T= _preprocess(episode_batch, batch_size_in_transitions, std=std, use_std=use_std)
        _random_log('using HERO sampler with step:{}, alpha:{}, and std: {}'.format(steps, alpha, use_std))
        if not no_her:
            future_ag, her_indexes = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T)
            transitions['g'][her_indexes] = future_ag

        # Re-compute reward since we may have substituted the goal.
        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g']) 

        ## model-based on-policy
        reward_list = [transitions['r']]
        last_state = transitions['o_2']
        if steps > 1:
            for _ in range(1, steps):
                state_array = last_state
                action_array = action_fun(o=state_array, g=transitions['g'])
                next_state_array = dynamic_model.predict_next_state(state_array, action_array)
               
                next_reward = _get_reward(obs_to_goal_fun(next_state_array), transitions['g'])
                reward_list.append(next_reward.copy())
                last_state = next_state_array

        last_Q = Q_fun(o=last_state, g=transitions['g'])
        target = 0
        for i in range(steps):
            target += pow(gamma, i) * reward_list[i]
        target += pow(gamma, steps) * last_Q.reshape(-1)
        transitions['r'] = target.copy()
        # allievate the model bias
        if steps > 1:
            target_step1 = reward_list[0] + gamma * Q_fun(o=transitions['o_2'], g=transitions['g']).reshape(-1)
            transitions['r'] = (alpha * transitions['r'] + target_step1) / (1 + alpha)
           
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)

    return _sample_her_transitions, _sample_hero_transitions, _sample_mbpo_transitions


