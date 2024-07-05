import gymnasium as gym

from agents.agentfactory import AgentFactory
from agents.tabularagent import TabularAgent
from util.metricstracker import MetricsTracker


def env_interaction(env: gym.Env, agent: TabularAgent, time_steps: int = 500, policy_type: str = 'behavior') -> float:
    '''
    Given an environment, agent, timesteps and a policy type performs a run in the environment and provides
    the actual return.
    If policy is behavior, an on-policy agent is being used
    If policy is target, an off-policy agent is being used
    '''
    obs, info = env.reset()
    actual_return = 0

    for _ in range(time_steps):
        old_obs = obs
        if policy_type == 'behavior':
            action = agent.behavior_policy(obs)
        if policy_type == 'target':
            action = agent.target_policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        actual_return += reward

        agent.update((old_obs, action, reward, obs))

        if terminated or truncated:
            # Episode ended.
            obs, info = env.reset()
            break
    env.close()
    return actual_return

def track_specific_agent(agent_id: str, episodes: int, time_steps_per_episode: int, env: gym.Env, policy_type: str):
    '''
    Tracks and plots the performance of an agent (Q-learning, SARSA, DQL, Random) for a number of episodes, time steps per episode,
    and on/off policy depending on policy_type (either behavior or target)
    '''
    tracker = MetricsTracker()
    agent = AgentFactory.create_agent(agent_id, env=env)
    for _ in range(episodes):   
        actual_return = env_interaction(env, agent, time_steps_per_episode, policy_type)
        tracker.record_return(agent_id, actual_return)
    tracker.plot(filename=str(agent_id) + str(policy_type)+'.png')
    
def track_all_simultaneously(episodes: int, time_steps_per_episode: int, env: gym.Env, policy_type: str, plot_title: str):
    '''
    Tracks and plots the performance of Q-learning, SARSA, DQL, Random for a number of episodes, time steps per episode,
    and on/off policy depending on policy_type (either behavior or target).
    This code is partially repeated (and a bit hard coded)from the function above but a tracker is needed for all 4
    agents at the same time so the above function cannot be used without adjustments.
    '''
    tracker = MetricsTracker()
    
    sarsa_id = "SARSA"
    sarsa_agent = AgentFactory.create_agent(sarsa_id, env=env)
    
    q_learning_id = "Q-LEARNING"
    q_learning_agent = AgentFactory.create_agent(q_learning_id, env=env)
    
    d_q_learning_id = "DOUBLE-Q-LEARNING"
    d_q_learning_agent = AgentFactory.create_agent(d_q_learning_id, env=env)
    
    random_id = "RANDOM"
    random_agent = AgentFactory.create_agent(random_id, env=env)
    
    for _ in range(episodes):   
        sarsa_return = env_interaction(env, sarsa_agent, time_steps_per_episode, policy_type)
        tracker.record_return(sarsa_id, sarsa_return)
        
        q_learning_return = env_interaction(env, q_learning_agent, time_steps_per_episode, policy_type)
        tracker.record_return(q_learning_id, q_learning_return)
        
        d_q_learning_return = env_interaction(env, d_q_learning_agent, time_steps_per_episode, policy_type)
        tracker.record_return(d_q_learning_id, d_q_learning_return)
        
        # Should hide this if you want to compare the other 3 actual agents, this is too distorted
        #random_return = env_interaction(env, random_agent, time_steps_per_episode, policy_type)
        #tracker.record_return(random_id, random_return)
        
    tracker.plot(filename=str(plot_title)+'.png', title=plot_title)

if __name__ == "__main__":
    env = gym.make("CliffWalking-v0", render_mode='human')
    '''track_sarsa(500, 500, env, 'behavior')
    track_sarsa(500, 500, env, 'target') #they're the same
    track_q_learning(500, 500, env, 'behavior')
    track_q_learning(500, 500, env, 'target')
    track_double_q_learning(500, 500, env, 'behavior')
    track_double_q_learning(500, 500, env, 'target')
    track_random(500, 500, env)'''
    track_all_simultaneously(500, 500, env, 'behavior', plot_title='Average Behavior Return, Exponential Decay Epsilon')
    track_all_simultaneously(500, 500, env, 'target', plot_title='Average Target Return, Exponential Decay Epsilon')
    
    

