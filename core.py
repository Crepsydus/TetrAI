import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import random
import colorama as cr

from game import Game

combs_standard = ["","tetris","tspin"]

device = torch.device("cuda")

class TetrisModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(256),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(512),
            nn.Flatten(),
        )

        self.extra_net = nn.Sequential(
            nn.Linear(37, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )

        self.cross_interaction = nn.Sequential(
            nn.Linear(512*11*5 + 256, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.output_layer = nn.Linear(256, 6)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        board = x[:, :230].view(x.size(0), 1, 23, 10)
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

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=1e-5)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = []
        self.memory_capacity = 512
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.005

    def act(self, state, epsilon, training=True):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)

        if not training:
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(state)
            self.model.train()
        else:
            if np.random.random() < epsilon:
                return np.random.randint(0, 6)
            q_values = self.model(state)

        return torch.argmax(q_values).item()

    def remember(self, transition):
        transition = (
            transition[0].cpu() if torch.is_tensor(transition[0]) else transition[0],
            transition[1].cpu() if torch.is_tensor(transition[1]) else transition[1],
            transition[2].cpu() if torch.is_tensor(transition[2]) else transition[2],
            transition[3].cpu() if torch.is_tensor(transition[3]) else transition[3],
            transition[4]
        )
        if len(self.memory) < self.memory_capacity:
            self.memory.append(transition)
        else:
            idx = np.random.randint(0, self.memory_capacity)
            self.memory[idx] = transition

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array([s.cpu().numpy() if torch.is_tensor(s) else s for s in states])).to(device)
        actions = torch.LongTensor(np.array([a.cpu().numpy() if torch.is_tensor(a) else a for a in actions])).to(device)
        rewards = torch.FloatTensor(np.array([r.cpu().numpy() if torch.is_tensor(r) else r for r in rewards])).to(device)
        next_states = torch.FloatTensor(np.array([ns.cpu().numpy() if torch.is_tensor(ns) else ns for ns in next_states])).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update_target_model()

    def soft_update_target_model(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

agent = Agent()
#%%
agent.model.load_state_dict(torch.load('model.pt', map_location=torch.device(device)))
#%%
episode_count = 500
move_limit = 150
empty_moves_penalty = -8
reward_gamma = 0.9

printer_cap = 10
replay_times = 1
autosave_cap = 20

epsilon_start = 1.0
epsilon_decay = 0.8 # multiplying
epsilon_refresh_cap = 20

epsilon = epsilon_start
mean_score = 200
scores = []
best_score = 0
g = Game()
for episode in range(episode_count):
    if episode % replay_times != 0 and episode != 0:
        g.reset(True)
    else:
        g.reset(False)
    g.create_piece()

    moves = 0
    move_chain = []

    while moves < move_limit and not g.lost:
        before_state = g.get_state()
        before_state_tensor = torch.FloatTensor(before_state).to(device).unsqueeze(0)
        before_score = g.score
        before_readiness = g.lines_readiness

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
            case 4:
                g.swap_hold()
            case 5:
                g.rotate()
        moves += 1
        reward = empty_moves_penalty
        if action == 3:
            reward = (g.score - before_score) - 13 + g.collisions + (g.lines_readiness - before_readiness)*100
        if g.lost:
            reward = -30
        if moves == move_limit:
            reward = -30

        if reward != empty_moves_penalty:
            new_move_chain = []
            local_gamma = reward_gamma
            for trajectory in move_chain[::-1]:
                new_move_chain.append((trajectory[0],trajectory[1],trajectory[2]+reward*local_gamma,
                                           trajectory[3],trajectory[4]))
                local_gamma *= reward_gamma
            move_chain = []
            new_move_chain = new_move_chain[::-1]
            new_move_chain.append((before_state_tensor, action, reward, g.get_state(), g.lost))
            for i in new_move_chain:
                agent.remember(i)
                # print(i[1],i[2])
            agent.replay()
            # print("----")
        else:
            move_chain.append((before_state_tensor, action, empty_moves_penalty, g.get_state(), g.lost))

    score = g.score
    print(f"{moves} ходов | эпсилон {epsilon:.3f} | {cr.Fore.GREEN}{score} очков {cr.Fore.RESET}")
    scores.append(score)
    if score > best_score:
        g.print_map()
        print(f"{cr.Fore.YELLOW}        Лучший за всё время{cr.Fore.RESET}")
        best_score = score
        torch.save(agent.model.state_dict(), 'best_score_model.pt')
    elif score > best_score * 0.85:
        g.print_map()
        print(f"{cr.Fore.BLUE}Достаточно крут{cr.Fore.RESET}")

    epsilon *= epsilon_decay

    if (episode+1) % printer_cap == 0 and episode != 0:
        g.print_map()
        print(f"Эпизод {episode}")
        mean_score = sum(scores) / len(scores)
        print(f"Средний счёт: {mean_score}")
        scores = []
    if (episode+1) % autosave_cap == 0:
        torch.save(agent.model.state_dict(), 'model.pt')
        print("AUTOSAVE")
    if (episode+1) % epsilon_refresh_cap == 0:
        epsilon = epsilon_start

print("Done")
#%%

torch.save(agent.model.state_dict(), 'model.pt')