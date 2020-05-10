import retro

def main():
	env = retro.make(game='CircusCharlie-Nes')

	print(env.action_space)
	#MultiBinary(9)
	print(env.observation_space)
	#Box(224, 240, 3)

	for i_episode in range(20):
		obs = env.reset()
		while True:
		#for t in range(1000):
			env.render()
			#print(obs)
			action = env.action_space.sample()
			obs, rew, done, info = env.step(action)
			print("Observation: ", obs)
			print("Image: ", obs.shape, "Reward: ",rew, "Done? ",done)
			print("Info ", info)
			if done:
				#print("Episode finished after {} timesteps".format(t+1))
				obs = env.reset()
	env.close()


if __name__ == "__main__":
	main()