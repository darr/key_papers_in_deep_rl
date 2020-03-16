# key_papers_in_deep_rl

What follows is a list of papers in deep RL that are worth reading. This is far from comprehensive, but should provide a useful starting point for someone looking to do research in the field.


content from:
```shell
https://spinningup.openai.com/en/latest/spinningup/keypapers.html
```

* [Model-Free RL](./Key_Papers_in_Deep_RL/Model-Free_RL)
* [Exploration](./Key_Papers_in_Deep_RL/Exploration)
* [Transfer and Multitask RL](./Key_Papers_in_Deep_RL/Transfer_and_Multitask_RL)
* [Hierarchy](./Key_Papers_in_Deep_RL/Hierarchy)
* [Memory](./Key_Papers_in_Deep_RL/Memory)
* [Model-Based RL](./Key_Papers_in_Deep_RL/Model-Based_RL)
* [Meta-RL](./Key_Papers_in_Deep_RL/Meta-RL)
* [Scaling RL](./Key_Papers_in_Deep_RL/Scaling_RL)
* [RL in the Real World](./Key_Papers_in_Deep_RL/RL_in_the_Real_World)
* [Safety](./Key_Papers_in_Deep_RL/Safety)
* [Imitation Learning and Inverse Reinforcement Learning](./Key_Papers_in_Deep_RL/Imitation_Learning_and_Inverse_Reinforcement_Learning)
* [Reproducibility, Analysis, and Critique](./Key_Papers_in_Deep_RL/Reproducibility_Analysis_and_Critique)
* [Bonus: Classic Papers in RL Theory or Review](./Key_Papers_in_Deep_RL/Bonus_Classic_Papers_in_RL_Theory_or_Review)

## 1.Model-Free RL

### a. Deep Q-Learning

* [1. Playing Atari with Deep Reinforcement Learning, Mnih et al, 2013. Algorithm: DQN. ](./Key_Papers_in_Deep_RL/Model-Free_RL/Deep_Q-Learning/Playing_Atari_with_Deep_Reinforcement_Learning.pdf)
   [(origin address)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [2. Deep Recurrent Q-Learning for Partially Observable MDPs, Hausknecht and Stone, 2015. Algorithm: Deep Recurrent Q-Learning.](./Key_Papers_in_Deep_RL/Model-Free_RL/Deep_Q-Learning/Deep_Recurrent_Q-Learning_for_Partially_Observable_MDPs.pdf)
   [(origin address)](https://arxiv.org/abs/1507.06527)
* [3. Dueling Network Architectures for Deep Reinforcement Learning, Wang et al, 2015. Algorithm: Dueling DQN.](./Key_Papers_in_Deep_RL/Model-Free_RL/Deep_Q-Learning/Dueling_Network_Architectures_for_Deep_Reinforcement_Learning.pdf)
   [(origin address)](https://arxiv.org/abs/1511.06581)
* [4. Deep Reinforcement Learning with Double Q-learning, Hasselt et al 2015. Algorithm: Double DQN.](./Key_Papers_in_Deep_RL/Model-Free_RL/Deep_Q-Learning/Deep_Reinforcement_Learning_with_Double_Q-learning.pdf)
   [(origin address)](https://arxiv.org/abs/1509.06461)
* [5. Prioritized Experience Replay, Schaul et al, 2015. Algorithm: Prioritized Experience Replay (PER).](./Key_Papers_in_Deep_RL/Model-Free_RL/Deep_Q-Learning/PRIORITIZED_EXPERIENCE_REPLAY.pdf)
   [(origin address)](https://arxiv.org/abs/1511.05952)
* [6. Rainbow: Combining Improvements in Deep Reinforcement Learning, Hessel et al, 2017. Algorithm: Rainbow DQN.](./Key_Papers_in_Deep_RL/Model-Free_RL/Deep_Q-Learning/Rainbow_Combining_Improvements_in_Deep_Reinforcement_Learning.pdf)
    [(origin address)](https://arxiv.org/abs/1710.02298)

### b. Policy Gradients

* [7. Asynchronous Methods for Deep Reinforcement Learning, Mnih et al, 2016. Algorithm: A3C.](./Key_Papers_in_Deep_RL/Model-Free_RL/Policy_Gradients/Asynchronous_Methods_for_Deep_Reinforcement_Learning.pdf)
    [(origin address)](https://arxiv.org/abs/1602.01783)
* [8. Trust Region Policy Optimization, Schulman et al, 2015. Algorithm: TRPO.](./Key_Papers_in_Deep_RL/Model-Free_RL/Policy_Gradients/Trust_Region_Policy_Optimization.pdf)
    [(origin address)](https://arxiv.org/abs/1502.05477)
* [9. High-Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al, 2015. Algorithm: GAE.](./Key_Papers_in_Deep_RL/Model-Free_RL/Policy_Gradients/HIGH-DIMENSIONAL_CONTINUOUS_CONTROL_USING_GENERALIZED_ADVANTAGE_ESTIMATION.pdf)
    [(origin address)](https://arxiv.org/abs/1506.02438)
* [10.    Proximal Policy Optimization Algorithms, Schulman et al, 2017. Algorithm: PPO-Clip, PPO-Penalty.](./Key_Papers_in_Deep_RL/Model-Free_RL/Policy_Gradients/Proximal_Policy_Optimization_Algorithms_1707.06347.pdf)
    [(origin address)](https://arxiv.org/abs/1707.06347)
* [11.    Emergence of Locomotion Behaviours in Rich Environments, Heess et al, 2017. Algorithm: PPO-Penalty.](./Key_Papers_in_Deep_RL/Model-Free_RL/Policy_Gradients/Emergence_of_Locomotion_Behaviours_in_Rich_Environments_1707.02286.pdf)
    [(origin address)](https://arxiv.org/abs/1707.02286)
* [12.    Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation, Wu et al, 2017. Algorithm: ACKTR.](./Key_Papers_in_Deep_RL/Model-Free_RL/Policy_Gradients/Scalable_trust-region_method_for_deep_reinforcement_learning_using_Kronecker-factored_approximation_1708.05144.pdf)
    [(origin address)](https://arxiv.org/abs/1708.05144)
* [13.    Sample Efficient Actor-Critic with Experience Replay, Wang et al, 2016. Algorithm: ACER.](./Key_Papers_in_Deep_RL/Model-Free_RL/Policy_Gradients/SAMPLE_EFFICIENT_ACTOR-CRITIC_WITH_EXPERIENCE_REPLAY_1611.01224.pdf)
    [(origin address)](https://arxiv.org/abs/1611.01224)

* [14.    Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, Haarnoja et al, 2018. Algorithm: SAC.](./Key_Papers_in_Deep_RL/Model-Free_RL/Policy_Gradients/Soft_Actor-Critic_Off-Policy_Maximum_Entropy_Deep_Reinforcement_Learning_with_a_Stochastic_Actor_1801.01290.pdf)
    [(origin address)](https://arxiv.org/abs/1801.01290)

### c. Deterministic Policy Gradients

* [15.    Deterministic Policy Gradient Algorithms, Silver et al, 2014. Algorithm: DPG.](./Key_Papers_in_Deep_RL/Model-Free_RL/Deterministic_Policy_Gradients/Deterministic_Policy_Gradient_Algorithms_14.pdf)
    [(origin address)](http://proceedings.mlr.press/v32/silver14.pdf)
* [16.    Continuous Control With Deep Reinforcement Learning, Lillicrap et al, 2015. Algorithm: DDPG.](./Key_Papers_in_Deep_RL/Model-Free_RL/Deterministic_Policy_Gradients/CONTINUOUS_CONTROL_WITH_DEEP_REINFORCEMENT_LEARNING_1509.02971.pdf)
    [(origin address)](https://arxiv.org/abs/1509.02971)
* [17.    Addressing Function Approximation Error in Actor-Critic Methods, Fujimoto et al, 2018. Algorithm: TD3.](./Key_Papers_in_Deep_RL/Model-Free_RL/Deterministic_Policy_Gradients/Addressing_Function_Approximation_Error_in_Actor-Critic_Methods_1802.09477.pdf)
    [(origin address)](https://arxiv.org/abs/1802.09477)

### d.Distributional RL

* [18.    A Distributional Perspective on Reinforcement Learning, Bellemare et al, 2017. Algorithm: C51.](./Key_Papers_in_Deep_RL/Model-Free_RL/Distributional_RL/A_Distributional_Perspective_on_Reinforcement_Learning_1707.06887.pdf)
    [(origin address)](https://arxiv.org/abs/1707.06887)
* [19     Distributional Reinforcement Learning with Quantile Regression, Dabney et al, 2017. Algorithm: QR-DQN.](./Key_Papers_in_Deep_RL/Model-Free_RL/Distributional_RL/Distributional_Reinforcement_Learning_with_Quantile_Regression_1710.10044.pdf)
    [(origin address)](https://arxiv.org/abs/1710.10044)
* [20.    Implicit Quantile Networks for Distributional Reinforcement Learning, Dabney et al, 2018. Algorithm: IQN.](./Key_Papers_in_Deep_RL/Model-Free_RL/Distributional_RL/Implicit_Quantile_Networks_for_Distributional_Reinforcement_Learning_1806.06923.pdf)
    [(origin address)](https://arxiv.org/abs/1806.06923)
* [21.    Dopamine: A Research Framework for Deep Reinforcement Learning, Anonymous, 2018. Contribution: Introduces Dopamine, a code repository containing implementations of DQN, C51, IQN, and Rainbow. ](./Key_Papers_in_Deep_RL/Model-Free_RL/Distributional_RL/dopamine_a_research_framework_for_deep_reinforcement_learning.pdf) [Code link.](https://github.com/google/dopamine)
    [(origin address)](https://openreview.net/forum?id=ByG_3s09KX)

### e. Policy Gradients with Action-Dependent Baselines

* [22.    Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic, Gu et al, 2016. Algorithm: Q-Prop.](./Key_Papers_in_Deep_RL/Model-Free_RL/Policy_Gradients_with_Action-Dependent_Baselines/Q-PROP_SAMPLE-EFFICIENT_POLICY_GRADIENT_WITH_AN_OFF-POLICY_CRITIC_1611.02247.pdf)
    [(origin address)](https://arxiv.org/abs/1611.02247)
* [23.    Action-depedent Control Variates for Policy Optimization via Stein’s Identity, Liu et al, 2017. Algorithm: Stein Control Variates.](./Key_Papers_in_Deep_RL/Model-Free_RL/Policy_Gradients_with_Action-Dependent_Baselines/ACTION-DEPENDENT_CONTROL_VARIATES_FOR_POLICY_OPTIMIZATION_VIA_STEINбпS_IDENTITY_1710.11198.pdf)
    [(origin address)](https://arxiv.org/abs/1710.11198)
* [24.    The Mirage of Action-Dependent Baselines in Reinforcement Learning, Tucker et al, 2018. Contribution: interestingly, critiques and reevaluates claims from earlier papers (including Q-Prop and stein control variates) and finds important methodological errors in them.](./Key_Papers_in_Deep_RL/Model-Free_RL/Policy_Gradients_with_Action-Dependent_Baselines/The_Mirage_of_Action-Dependent_Baselines_in_Reinforcement_Learning_1802.10031.pdf)
    [(origin address)](https://arxiv.org/abs/1802.10031)

### f. Path-Consistency Learning

* [25.    Bridging the Gap Between Value and Policy Based Reinforcement Learning, Nachum et al, 2017. Algorithm: PCL.](./Key_Papers_in_Deep_RL/Model-Free_RL/Path-Consistency_Learning/Bridging_the_Gap_Between_Value_and_Policy_Based_Reinforcement_Learning_1702.08892.pdf)
    [(origin address)](https://arxiv.org/abs/1702.08892)
* [26.    Trust-PCL: An Off-Policy Trust Region Method for Continuous Control, Nachum et al, 2017. Algorithm: Trust-PCL.](./Key_Papers_in_Deep_RL/Model-Free_RL/Path-Consistency_Learning/TRUST-PCL_AN_OFF-POLICY_TRUST_REGION_METHOD_FOR_CONTINUOUS_CONTROL_1707.01891.pdf)
    [(origin address)](https://arxiv.org/abs/1707.01891)

### g. Other Directions for Combining Policy-Learning and Q-learning

* [27.    Combining Policy Gradient and Q-learning, O’Donoghue et al, 2016. Algorithm: PGQL.](./Key_Papers_in_Deep_RL/Model-Free_RL/Other_Directions_for_Combining_Policy-Learning_and_Q-Learning/COMBINING_POLICY_GRADIENT_AND_Q-LEARNING_1611.01626.pdf)
    [(origin address)](https://arxiv.org/abs/1611.01626)
* [28.    The Reactor: A Fast and Sample-Efficient Actor-Critic Agent for Reinforcement Learning, Gruslys et al, 2017. Algorithm: Reactor.](./Key_Papers_in_Deep_RL/Model-Free_RL/Other_Directions_for_Combining_Policy-Learning_and_Q-Learning/THE_REACTOR_A_FAST_AND_SAMPLE-EFFICIENT_ACTOR-CRITIC_AGENT_FOR_REINFORCEMENT_LEARNING_1704.04651.pdf)
    [(origin address)](https://arxiv.org/abs/1704.04651)
* [29.    Interpolated Policy Gradient: Merging On-Policy and Off-Policy Gradient Estimation for Deep Reinforcement Learning, Gu et al, 2017. Algorithm: IPG.](./Key_Papers_in_Deep_RL/Model-Free_RL/Other_Directions_for_Combining_Policy-Learning_and_Q-Learning/interpolated-policy-gradient-merging-on-policy-and-off-policy-gradient-estimation-for-deep-reinforcement-learning.pdf)
    [(origin address)](http://papers.nips.cc/paper/6974-interpolated-policy-gradient-merging-on-policy-and-off-policy-gradient-estimation-for-deep-reinforcement-learning)
* [30.    Equivalence Between Policy Gradients and Soft Q-Learning, Schulman et al, 2017. Contribution: Reveals a theoretical link between these two families of RL algorithms.](./Key_Papers_in_Deep_RL/Model-Free_RL/Other_Directions_for_Combining_Policy-Learning_and_Q-Learning/Equivalence_Between_Policy_Gradients_and_Soft_Q-Learning_1704.06440.pdf)
    [(origin address)](https://arxiv.org/abs/1704.06440)

### h. Evolutionary Algorithms

* [31.    Evolution Strategies as a Scalable Alternative to Reinforcement Learning, Salimans et al, 2017. Algorithm: ES.](./Key_Papers_in_Deep_RL/Model-Free_RL/Evolutionary_Algorithms/Evolution_Strategies_as_a_Scalable_Alternative_to_Reinforcement_Learning_1703.03864.pdf)
    [(origin address)](https://arxiv.org/abs/1703.03864)


## 2. Exploration

### a. Intrinsic Motivation

* [32.    VIME: Variational Information Maximizing Exploration, Houthooft et al, 2016. Algorithm: VIME.](./Key_Papers_in_Deep_RL/Exploration/Intrinsic_Motivation/VIME_Variational_Information_Maximizing_Exploration_1605.09674.pdf)
    [(origin address)](https://arxiv.org/abs/1605.09674)
* [33.    Unifying Count-Based Exploration and Intrinsic Motivation, Bellemare et al, 2016. Algorithm: CTS-based Pseudocounts.](./Key_Papers_in_Deep_RL/Exploration/Intrinsic_Motivation/Unifying_Count-Based_Exploration_and_Intrinsic_Motivation_1606.01868.pdf)
    [(origin address)](https://arxiv.org/abs/1606.01868)
* [34.    Count-Based Exploration with Neural Density Models, Ostrovski et al, 2017. Algorithm: PixelCNN-based Pseudocounts.](./Key_Papers_in_Deep_RL/Exploration/Intrinsic_Motivation/Count-Based_Exploration_with_Neural_Density_Models_1703.01310.pdf)
    [(origin address)](https://arxiv.org/abs/1703.01310)
* [35.    #Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning, Tang et al, 2016. Algorithm: Hash-based Counts.](./Key_Papers_in_Deep_RL/Exploration/Intrinsic_Motivation/Exploration_A_Study_of_Count-Based_Exploration_for_Deep_Reinforcement_Learning_1611.04717.pdf)
    [(origin address)](https://arxiv.org/abs/1611.04717)

* [36.    EX2: Exploration with Exemplar Models for Deep Reinforcement Learning, Fu et al, 2017. Algorithm: EX2.](./Key_Papers_in_Deep_RL/Exploration/Intrinsic_Motivation/EX2_Exploration_with_Exemplar_Models_for_Deep_Reinforcement_Learning_1703.01260.pdf)
    [(origin address)](https://arxiv.org/abs/1703.01260)
* [37.    Curiosity-driven Exploration by Self-supervised Prediction, Pathak et al, 2017. Algorithm: Intrinsic Curiosity Module (ICM).](./Key_Papers_in_Deep_RL/Exploration/Intrinsic_Motivation/Curiosity-driven_Exploration_by_Self-supervised_Prediction_1705.05363.pdf)
    [(origin address)](https://arxiv.org/abs/1705.05363)
* [38.    Large-Scale Study of Curiosity-Driven Learning, Burda et al, 2018. Contribution: Systematic analysis of how surprisal-based intrinsic motivation performs in a wide variety of environments.](./Key_Papers_in_Deep_RL/Exploration/Intrinsic_Motivation/Large-Scale_Study_of_Curiosity-Driven_Learning_1808.04355.pdf)
    [(origin address)](https://arxiv.org/abs/1808.04355)
* [39.    Exploration by Random Network Distillation, Burda et al, 2018. Algorithm: RND.](./Key_Papers_in_Deep_RL/Exploration/Intrinsic_Motivation/EXPLORATION_BY_RANDOM_NETWORK_DISTILLATION_1810.12894.pdf)
    [(origin address)](https://arxiv.org/abs/1810.12894)

### b. Unsupervised RL

* [40.    Variational Intrinsic Control, Gregor et al, 2016. Algorithm: VIC.](./Key_Papers_in_Deep_RL/Exploration/Unsupervised_RL/VARIATIONAL_INTRINSIC_CONTROL_1611.07507.pdf)
    [(origin address)](https://arxiv.org/abs/1611.07507)
* [41.    Diversity is All You Need: Learning Skills without a Reward Function, Eysenbach et al, 2018. Algorithm: DIAYN.](./Key_Papers_in_Deep_RL/Exploration/Unsupervised_RL/DIVERSITY_IS_ALL_YOU_NEED_LEARNING_SKILLS_WITHOUT_A_REWARD_FUNCTION_1802.06070.pdf)
    [(origin address)](https://arxiv.org/abs/1802.06070)
* [42.    Variational Option Discovery Algorithms, Achiam et al, 2018. Algorithm: VALOR.](./Key_Papers_in_Deep_RL/Exploration/Unsupervised_RL/Variational_Option_Discovery_Algorithms_1807.10299.pdf)
    [(origin address)](https://arxiv.org/abs/1807.10299)

## 3. Transfer and Multitask RL

* [43.    Progressive Neural Networks, Rusu et al, 2016. Algorithm: Progressive Networks.](./Key_Papers_in_Deep_RL/Transfer_and_Multitask_RL/Progressive_Neural_Networks_1606.04671.pdf)
    [(origin address)](https://arxiv.org/abs/1606.04671)
* [44.    Universal Value Function Approximators, Schaul et al, 2015. Algorithm: UVFA.](./Key_Papers_in_Deep_RL/Transfer_and_Multitask_RL/Universal_Value_Function_Approximators_schaul15.pdf)
    [(origin address)](http://proceedings.mlr.press/v37/schaul15.pdf)
* [45.    Reinforcement Learning with Unsupervised Auxiliary Tasks, Jaderberg et al, 2016. Algorithm: UNREAL.](./Key_Papers_in_Deep_RL/Transfer_and_Multitask_RL/REINFORCEMENT_LEARNING_WITH_UNSUPERVISED_AUXILIARY_TASKS_1611.05397.pdf)
    [(origin address)](https://arxiv.org/abs/1611.05397)
* [46.    The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously, Cabi et al, 2017. Algorithm: IU Agent.](./Key_Papers_in_Deep_RL/Transfer_and_Multitask_RL/The_Intentional_Unintentional_Agent_Learning_to_Solve_Many_Continuous_Control_Tasks_Simultaneously_1707.03300.pdf)
    [(origin address)](https://arxiv.org/abs/1707.03300)
* [47.    PathNet: Evolution Channels Gradient Descent in Super Neural Networks, Fernando et al, 2017. Algorithm: PathNet.](./Key_Papers_in_Deep_RL/Transfer_and_Multitask_RL/PathNet_Evolution_Channels_Gradient_Descent_in_Super_Neoural_Networks_1701.08734.pdf)
    [(origin address)](https://arxiv.org/abs/1701.08734)
* [48.    Mutual Alignment Transfer Learning, Wulfmeier et al, 2017. Algorithm: MATL.](./Key_Papers_in_Deep_RL/Transfer_and_Multitask_RL/Mutual_Alignment_Transfer_Learning_1707.07907.pdf)
    [(origin address)](https://arxiv.org/abs/1707.07907)
* [49.    Learning an Embedding Space for Transferable Robot Skills, Hausman et al, 2018.](./Key_Papers_in_Deep_RL/Transfer_and_Multitask_RL/learning_an_embedding_space_for_transferable_robot_skills.pdf)
    [(origin address)](https://openreview.net/forum?id=rk07ZXZRb&noteId=rk07ZXZRb)
* [50.    Hindsight Experience Replay, Andrychowicz et al, 2017. Algorithm: Hindsight Experience Replay (HER).](./Key_Papers_in_Deep_RL/Transfer_and_Multitask_RL/Hindsight_Experience_Replay_1707.01495.pdf)
    [(origin address)](https://arxiv.org/abs/1707.01495)

## 4. Hierarchy

* [51.    Strategic Attentive Writer for Learning Macro-Actions, Vezhnevets et al, 2016. Algorithm: STRAW.](./Key_Papers_in_Deep_RL/Hierarchy/Strategic_Attentive_Writer_for_Learning_Macro-Actions_1606.04695.pdf)
    [(origin address)](https://arxiv.org/abs/1606.04695)
* [52.    FeUdal Networks for Hierarchical Reinforcement Learning, Vezhnevets et al, 2017. Algorithm: Feudal Networks](./Key_Papers_in_Deep_RL/Hierarchy/FeUdal_Networks_for_Hierarchical_Reinforcement_Learning_1703.01161.pdf)
    [(origin address)](https://arxiv.org/abs/1703.01161)
* [53.    Data-Efficient Hierarchical Reinforcement Learning, Nachum et al, 2018. Algorithm: HIRO.](./Key_Papers_in_Deep_RL/Hierarchy/Data-Efficient_Hierarchical_Reinforcement_Learning_1805.08296.pdf)
    [(origin address)](https://arxiv.org/abs/1805.08296)

## 5. Memory

* [54.    Model-Free Episodic Control, Blundell et al, 2016. Algorithm: MFEC.](./Key_Papers_in_Deep_RL/Memory/Model-Free_Episodic_Control_1606.04460.pdf)
    [(origin address)](https://arxiv.org/abs/1606.04460)
* [55.    Neural Episodic Control, Pritzel et al, 2017. Algorithm: NEC.](./Key_Papers_in_Deep_RL/Memory/Neural_Episodic_Control_1703.01988.pdf)
    [(origin address)](https://arxiv.org/abs/1703.01988)
* [56.    Neural Map: Structured Memory for Deep Reinforcement Learning, Parisotto and Salakhutdinov, 2017. Algorithm: Neural Map.](./Key_Papers_in_Deep_RL/Memory/NEURAL_MAP_STRUCTURED_MEMORY_FOR_DEEP_REINFORCEMENT_LEARNING_1702.08360.pdf)
    [(origin address)](https://arxiv.org/abs/1702.08360)
* [57.    Unsupervised Predictive Memory in a Goal-Directed Agent, Wayne et al, 2018. Algorithm: MERLIN.](./Key_Papers_in_Deep_RL/Memory/Unsupervised_Predictive_Memory_in_a_Goal-Directed_Agent_1803.10760.pdf)
    [(origin address)](https://arxiv.org/abs/1803.10760)
* [58.    Relational Recurrent Neural Networks, Santoro et al, 2018. Algorithm: RMC.](./Key_Papers_in_Deep_RL/Memory/Relational_recurrent_neural_networks_1806.01822.pdf)
    [(origin address)](https://arxiv.org/abs/1806.01822)

## 6. Model-Based RL

### a. Model is Learned

* [59.    Imagination-Augmented Agents for Deep Reinforcement Learning, Weber et al, 2017. Algorithm: I2A.](./Key_Papers_in_Deep_RL/Model-Based_RL/Model_is_Learned/Imagination-Augmented_Agents_for_Deep_Reinforcement_Learning_1707.06203.pdf)
    [(origin address)](https://arxiv.org/abs/1707.06203)
* [60.   Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning, Nagabandi et al, 2017. Algorithm: MBMF.](./Key_Papers_in_Deep_RL/Model-Based_RL/Neural_Network_Dynamics_for_Model-Based_Deep_Reinforcement_Learning_with_Model-Free_Fine-Tuning_1708.02596.pdf)
    [(origin address)](https://arxiv.org/abs/1708.02596)
* [61.    Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning, Feinberg et al, 2018. Algorithm: MVE.](./Key_Papers_in_Deep_RL/Model-Based_RL/Model_is_Learned/Model-Based_Value_Expansion_for_Efficient_Model-Free_Reinforcement_Learning_1803.00101.pdf)
    [(origin address)](https://arxiv.org/abs/1803.00101)
* [62.    Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion, Buckman et al, 2018. Algorithm: STEVE.](./Key_Papers_in_Deep_RL/Model-Based_RL/Model_is_Learned/Sample-Efficient_Reinforcement_Learning_with_Stochastic_Ensemble_Value_Expansion_1807.01675.pdf)
    [(origin address)](https://arxiv.org/abs/1807.01675)
* [63.    Model-Ensemble Trust-Region Policy Optimization, Kurutach et al, 2018. Algorithm: ME-TRPO.](./Key_Papers_in_Deep_RL/Model-Based_RL/Model_is_Learned/model_ensemble_trust_region_policy_optimization.pdf)
    [(origin address)](https://openreview.net/forum?id=SJJinbWRZ&noteId=SJJinbWRZ)
* [64.    Model-Based Reinforcement Learning via Meta-Policy Optimization, Clavera et al, 2018. Algorithm: MB-MPO.](./Key_Papers_in_Deep_RL/Model-Based_RL/Model_is_Learned/Model-Based_Reinforcement_Learning_via_Meta-Policy_Optimization_1809.05214.pdf)
    [(origin address)](https://arxiv.org/abs/1809.05214)
* [65.    Recurrent World Models Facilitate Policy Evolution, Ha and Schmidhuber, 2018.](./Key_Papers_in_Deep_RL/Model-Based_RL/Model_is_Learned/Recurrent_World_Models_Facilitate_Policy_Evolution_1809.01999.pdf)
    [(origin address)](https://arxiv.org/abs/1809.01999)

### b. Model is Given

* [66]    Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm, Silver et al, 2017. Algorithm: AlphaZero.
* [67]    Thinking Fast and Slow with Deep Learning and Tree Search, Anthony et al, 2017. Algorithm: ExIt.


## 7. Meta-RL

* [68]    RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning, Duan et al, 2016. Algorithm: RL^2.
* [69]    Learning to Reinforcement Learn, Wang et al, 2016.
* [70]    Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks, Finn et al, 2017. Algorithm: MAML.
* [71]    A Simple Neural Attentive Meta-Learner, Mishra et al, 2018. Algorithm: SNAIL.


## 8. Scaling RL

* [72]    Accelerated Methods for Deep Reinforcement Learning, Stooke and Abbeel, 2018. Contribution: Systematic analysis of parallelization in deep RL across algorithms.
* [73]    IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures, Espeholt et al, 2018. Algorithm: IMPALA.
* [74]    Distributed Prioritized Experience Replay, Horgan et al, 2018. Algorithm: Ape-X.
* [75]    Recurrent Experience Replay in Distributed Reinforcement Learning, Anonymous, 2018. Algorithm: R2D2.
* [76]    RLlib: Abstractions for Distributed Reinforcement Learning, Liang et al, 2017. Contribution: A scalable library of RL algorithm implementations. Documentation link.

## 9. RL in the Real World

* [77]    Benchmarking Reinforcement Learning Algorithms on Real-World Robots, Mahmood et al, 2018.
* [78]    Learning Dexterous In-Hand Manipulation, OpenAI, 2018.
* [79]    QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation, Kalashnikov et al, 2018. Algorithm: QT-Opt.
* [80]    Horizon: Facebook’s Open Source Applied Reinforcement Learning Platform, Gauci et al, 2018.


## 10. Safety

* [81]    Concrete Problems in AI Safety, Amodei et al, 2016. Contribution: establishes a taxonomy of safety problems, serving as an important jumping-off point for future research. We need to solve these!
* [82]    Deep Reinforcement Learning From Human Preferences, Christiano et al, 2017. Algorithm: LFP.
* [83]    Constrained Policy Optimization, Achiam et al, 2017. Algorithm: CPO.
* [84]    Safe Exploration in Continuous Action Spaces, Dalal et al, 2018. Algorithm: DDPG+Safety Layer.
* [85]    Trial without Error: Towards Safe Reinforcement Learning via Human Intervention, Saunders et al, 2017. Algorithm: HIRL.
* [86]    Leave No Trace: Learning to Reset for Safe and Autonomous Reinforcement Learning, Eysenbach et al, 2017. Algorithm: Leave No Trace.

## 11. Imitation Learning and Inverse Reinforcement Learning

* [87]    Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy, Ziebart 2010. Contributions: Crisp formulation of maximum entropy IRL.
* [88]    Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization, Finn et al, 2016. Algorithm: GCL.
* [89]    Generative Adversarial Imitation Learning, Ho and Ermon, 2016. Algorithm: GAIL.
* [90]    DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills, Peng et al, 2018. Algorithm: DeepMimic.
* [91]    Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow, Peng et al, 2018. Algorithm: VAIL.
* [92]    One-Shot High-Fidelity Imitation: Training Large-Scale Deep Nets with RL, Le Paine et al, 2018. Algorithm: MetaMimic.

## 12. Reproducibility, Analysis, and Critique

* [93]    Benchmarking Deep Reinforcement Learning for Continuous Control, Duan et al, 2016. Contribution: rllab.
* [94]    Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control, Islam et al, 2017.
* [95]    Deep Reinforcement Learning that Matters, Henderson et al, 2017.
* [96]    Where Did My Optimum Go?: An Empirical Analysis of Gradient Descent Optimization in Policy Gradient Methods, Henderson et al, 2018.
* [97]    Are Deep Policy Gradient Algorithms Truly Policy Gradient Algorithms?, Ilyas et al, 2018.
* [98]    Simple Random Search Provides a Competitive Approach to Reinforcement Learning, Mania et al, 2018.
* [99]    Benchmarking Model-Based Reinforcement Learning, Wang et al, 2019.


## 13. Bonus:Classic Papers in RL Theory or Review

* [100]   Policy Gradient Methods for Reinforcement Learning with Function Approximation, Sutton et al, 2000. Contributions: Established policy gradient theorem and showed convergence of policy gradient algorithm for arbitrary policy classes.
* [101]   An Analysis of Temporal-Difference Learning with Function Approximation, Tsitsiklis and Van Roy, 1997. Contributions: Variety of convergence results and counter-examples for value-learning methods in RL.
* [102]   Reinforcement Learning of Motor Skills with Policy Gradients, Peters and Schaal, 2008. Contributions: Thorough review of policy gradient methods at the time, many of which are still serviceable descriptions of deep RL methods.
* [103]   Approximately Optimal Approximate Reinforcement Learning, Kakade and Langford, 2002. Contributions: Early roots for monotonic improvement theory, later leading to theoretical justification for TRPO and other algorithms.
* [104]   A Natural Policy Gradient, Kakade, 2002. Contributions: Brought natural gradients into RL, later leading to TRPO, ACKTR, and several other methods in deep RL.
* [105]   Algorithms for Reinforcement Learning, Szepesvari, 2009. Contributions: Unbeatable reference on RL before deep RL, containing foundations and theoretical background.
