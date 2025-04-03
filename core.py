import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import colorama as cr

from game import Game

standard = ["i", "o", "t", "s", "z", "l", "j", ""]
combs_standard = ["","tetris","tspin"]

device = torch.device("cuda")

class TetrisModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(), # channels * 11 * 5
        )

        self.extra_net = nn.Sequential(
            nn.Linear(17, 128),
            nn.ReLU(),
        )


        self.cross_interaction = nn.Sequential(
            nn.Linear(256 * 11 * 5 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.output_layer = nn.Linear(256, 6)

    def forward(self, x):
        batch_size = x.size(0)
        board = x[:, :230].view(batch_size, 1, 23, 10)
        extra = x[:, 230:]

        board_features = self.conv_net(board)
        extra_features = self.extra_net(extra)
        combined = torch.cat([board_features, extra_features], dim=1)
        features = self.cross_interaction(combined)

        return self.output_layer(features)

class Agent:
    def __init__(self):
        self.model = TetrisModel().to(device)
        self.target_model = TetrisModel().to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.buffer = []
        self.memory = []
        self.memory_capacity = 10000
        self.batch_size = 128
        self.gamma = 0.95

    def act(self, state, epsilon=0.25):
        if np.random.random() < epsilon:
            return np.random.randint(0, 5)
        else:
            with torch.no_grad():
                q_values = self.model(state)
                return torch.argmax(q_values).item()

    def remember(self):
        self.memory = self.memory + self.buffer
        self.buffer = []

    def replay(self):
        random_batch = random.sample(self.memory, self.batch_size//2)
        self.memory = self.memory + self.buffer
        if len(self.memory) > self.memory_capacity:
            self.memory = random.sample(self.memory, self.memory_capacity)
        batch = random_batch + random.sample(self.buffer, self.batch_size//2)
        states, actions, rewards, next_states = zip(*batch)
        self.buffer = []

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + self.gamma * next_q

        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

agent = Agent()
#%%
agent.model.load_state_dict(torch.load('model_weights.pt', map_location=torch.device(device)))
#%%
episode_count = 1500
move_limit = 500

printer_cap = 10
replay_times = 10
epsilon_start = 0.75
epsilon = epsilon_start
epsilon_decay = epsilon / replay_times

autosave_cap = 200

scores = []
g = Game()
for episode in range(episode_count):
    if episode % replay_times != 0 and episode != 0:
        g.reset(True)
    else:
        g.reset(False)
        epsilon = epsilon_start
    g.create_piece()

    moves = 0
    trajectories = []

    while (moves < move_limit and not g.lost and move_limit) or (not move_limit and not g.lost):
        before_state = g.get_state()
        before_score = g.score
        action = agent.act(torch.FloatTensor(before_state).to(device).unsqueeze(0), epsilon)
        match int(action):
            case 0:
                g.move_left()
            case 1:
                g.move_right()
            case 2:
                g.soft_drop()
            case 3:
                g.drop()
                prev_positions = []
            case 4:
                g.swap_hold()
            case 5:
                g.rotate()
        moves += 1
        score_difference = g.score - before_score
        reward = -4
        if score_difference > 0:
            reward = score_difference - 15 + g.collisions
        if g.lost:
            reward = -15

        trajectories.append([before_state, action, reward, g.get_state()])
    epsilon -= epsilon_decay

    # дисконтирование
    temp_trajectories = []
    i = -1
    local_gamma = 0.85
    last_reward = 0
    while -i <= len(trajectories):
        tr = trajectories[i]
        if tr[2] != -4:
            last_reward = tr[2]
            local_gamma = 0.9
        else:
            tr[2] = tr[2] + last_reward * local_gamma * 0.6
            local_gamma *= local_gamma
        i -= 1
        temp_trajectories.append(tr)
    trajectories = [tuple(tr) for tr in reversed(temp_trajectories)]
    agent.buffer = trajectories

    if len(agent.memory) >= agent.batch_size//2 and len(trajectories) >= agent.batch_size//2:
        agent.replay()
    else:
        agent.remember()

    score = g.score
    print(f"{moves} ходов | {cr.Fore.GREEN}{score} очков {cr.Fore.RESET}")
    scores.append(score)

    if (episode+1) % printer_cap == 0 and episode != 0:
        g.print_map()
        print(f"Эпизод {episode}")
        mean_score = sum(scores) / len(scores)
        print(f"Средний счёт: {mean_score}")
        scores = []
    if (episode+1) % 10 == 0:
        agent.update_target_model()
    if (episode+1) % autosave_cap == 0:
        torch.save(agent.model.state_dict(), 'model_weights.pt')
        print("AUTOSAVE")

print("Done")
#%%

torch.save(agent.model.state_dict(), 'model_weights.pt')