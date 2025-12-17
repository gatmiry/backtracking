# Simple task in Turi Bolt

This example shows a minimal Turi Bolt task example. It demonstrates the following aspects:

- The task's configuration file.
- The task's setup file to install dependencies.
- The task's code, including various Bolt APIs, like updating custom status messages and sending metrics.

## Execution

1. Follow the instructions in the [Bolt docs](https://bolt.apple.com/docs/get-started.html) to install the Turi Bolt client.
2. Submit the task: 
    ```
    cd bolt-sample
    bolt task submit --tar simple --config simple/config.yaml --max-retries 3
    ```
     
     (Setting --max-retries will automatically retry the task submission the specified number of times in case of failure. This highly recommended option makes task submission resilient to unexpected node failures and temporary network/service outages. For more details, see our [automatic retries documentation](https://bolt.apple.com/docs/tasks.html#retrying-a-task))
    
3. Log into the web UI using the "Live dashboard" link given when submitting the task.
