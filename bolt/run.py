import os
import random
import time

# The names package was made available by installing it in the setup command.
#import names

# The apple-bolt package is available on all Bolt tasks.
import apple_bolt as bolt
import torch
import transformers
from transformers import AutoModel
# The API lets you access the config file programmatically.
# This is useful for retrieving parameters or, as seen in more complex examples,
# constructing new configs for launching children.
config = bolt.get_current_config()
random.seed(bolt.get_current_config()['parameters']['random_seed'])

# Status messages are great for coarse-grained progress tracking. They appear
# in the task details at runtime (both CLI and Web UI).
bolt.set_status_message("Starting")

messages = []
results = []
for i in range(1, 20):
    time.sleep(5)
    message = 'Iteration %s' % str(i)
    print(message)
    messages.append(message)
    results.append(i ** 2)
    a = torch.tensor([1,2,4])
    print('a is ', a)
    # Bolt supports a variety of different metric types. Metrics are
    # automatically visualized on the task overview page and are available
    # for ad-hoc querying via the API.
    bolt.send_metrics({
        'Progress': i,
        'Random Accuracy': random.random(),
        'List Value': [random.random() for i in range(i)],
        'Progress Text': message,
        'Accumulated Progress Text': messages
    })

    mymodel = AutoModel.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# All files written to the artifacts directory are persisted synced to Blobby.
# They are accessible after task completion by all members of the task
# viewers group.
with open(os.path.join(bolt.ARTIFACT_DIR, 'results.txt'), 'w') as f:
    f.write(str(results))

bolt.set_status_message('Done')
print('I am done!')
