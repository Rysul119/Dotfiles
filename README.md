#dotfiles


My Ubuntu and Mac terminal config dot files for customizing z shell using PowerLevel10k theme and Oh My Zsh.\

score = 0
steps = 1000
gamma = 0.98
for e in range(episodes):
	s = env.reset()
        ep_memory = []
        actions = []
        logits = []
        history = []
        ep_score = 0
        done = False
        # with tf.GradientTape() as tape:
        for step in range(steps):
            s = s.reshape([1, 4])
            act_prob = model(s)
            act = np.random.choice(range(env.action_space.n), p=act_prob.numpy()[0])
            s, r, done, _ = env.step(act)
            history.append((s, act, r))
            score += r
            if done:
                R = 0
                for s, act, r in history[::-1]:
                    R = r + gamma * R
                    with tf.GradientTape() as tape:
                        output = model(s)
                        actProbs = tf.gather(tf.reshape(output, [-1]), a)
                        loss = -tf.reduce_mean(tf.math.log(actProbs) * R)
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                break
