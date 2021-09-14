from gym.envs.registration import register
register(
    id='CartPoleBullet-v2',
    entry_point='gym_transfer.envs:CartPoleBulletEnv',
    max_episode_steps=75,
    reward_threshold=475.0,
)
register(
    id='CartPoleBulletPO-v2',
    entry_point='gym_transfer.envs:CartPoleBulletPOEnv',
    max_episode_steps=75,
    reward_threshold=475.0,
)

register(
    id='CartPoleBulletPOScaled-v2',
    entry_point='gym_transfer.envs:CartPoleBulletPOScaledEnv',
    max_episode_steps=75,
    reward_threshold=475.0,
)

register(
    id='MountainCarScaled-v0',
    entry_point='gym_transfer.envs:MountainCarEnvScaled',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='AcrobotSparse-v1',
    entry_point='gym_transfer.envs:AcrobotEnvSparse',
#     reward_threshold=-100.0,
    max_episode_steps=500,
)

register(
    id='AcrobotDense-v1',
    entry_point='gym_transfer.envs:AcrobotEnvDense',
#     reward_threshold=-100.0,
    max_episode_steps=500,
)
