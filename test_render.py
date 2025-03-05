import ogbench
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import time 

# Make an environment and datasets (they will be automatically downloaded).
dataset_name = 'visual-antmaze-medium-navigate-v0'
# env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name, env_only=True)
env = ogbench.make_env_and_datasets(dataset_name, env_only=True)

# Train your offline goal-conditioned RL agent on the dataset.
# ...
class CNNBlock(nn.Module):
    def __init__(self, inp, out, dropout=0.1, batch_norm=False):
        super(CNNBlock, self).__init__()
        if batch_norm:
            self.model = nn.Sequential(
                    nn.Conv2d(inp, out, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.BatchNorm2d(out),
                    nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
                )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
            )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.model(x)
    
class Encoder(nn.Module):
    def __init__(self, inp_channel=3, filters=[32, 32, 32], dropout=0.1, image_size=(3, 64, 64), 
                 batch_norm=False, layer_norm=True, out_emb = 128):
        super(Encoder, self).__init__()
        self.inp_channel = inp_channel 
        self.filters = [inp_channel] + filters
        self.layer_norm = layer_norm
        self.image_size = list(image_size)
        self.stack_blocks = nn.ModuleList([
                        CNNBlock(self.filters[i], self.filters[i+1], batch_norm=batch_norm)
                        for i in range(len(self.filters)-1)
                        ])
        ln_size = self._inp_size()
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.layer_nm = nn.LayerNorm(ln_size)
        self.mlp = nn.Linear(np.prod(list(ln_size)), out_emb)

    def forward(self, x):
        for layer in self.stack_blocks:
            x = layer(x)
        if self.layer_norm:
            x = self.layer_nm(x)
        x = x.reshape(x.shape[0], -1)
        x = self.mlp(x)
        return x
    
    def _inp_size(self):
        batch_size = 1
        x = torch.zeros([batch_size] + self.image_size)
        for layer in self.stack_blocks:
            x = layer(x)
        return x.shape[1:]

# Evaluate the agent.
encoder = Encoder(out_emb=8)
for ep in range(1):
    steps = time.time()
    for task_id in [1, 2, 3, 4, 5]:
        # Reset the environment and set the evaluation task.
        ob, info = env.reset(
            options=dict(
                task_id=task_id,  # Set the evaluation task. Each environment provides five
                                # evaluation goals, and `task_id` must be in [1, 5].
                # render_goal=True,  # Set to `True` to get a rendered goal image (optional).
            )
        )

        goal = info['goal']  # Get the goal observation to pass to the agent.
        # goal_rendered = info['goal_rendered']  # Get the rendered goal image (optional).

        done = False
        while not done:
            action = env.action_space.sample()  # Replace this with your agent's action.
            acs = encoder(torch.tensor(ob.transpose(2, 0, 1)/255.0, dtype=torch.float).unsqueeze(0))
            ob, reward, terminated, truncated, info = env.step(action)  # Gymnasium-style step.
            ob_shape = ob.shape
            # If the agent reaches the goal, `terminated` will be `True`. If the episode length
            # exceeds the maximum length without reaching the goal, `truncated` will be `True`.
            # `reward` is 1 if the agent reaches the goal and 0 otherwise.
            done = terminated or truncated
            # frame = env.render()  # Render the current frame (optional).
            # steps += 1
        success = info['success']
    print(time.time()- steps)
print(ob_shape)
print(acs.shape)
# print(frame.shape)