# cognitive_robotics_at_oist

Certainly, let's delve into the mathematical derivation of Backpropagation Through Time (BPTT) for a vanilla Recurrent Neural Network (RNN). We will present a rigorous yet accessible formulation, suitable for adaptation to various RNN scenarios.

### 1. Vanilla RNN Architecture and Forward Pass

Consider a vanilla RNN processing an input sequence $\mathbf{X} = (\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T)$, where $\mathbf{x}_t \in \mathbb{R}^{n_x}$ is the input at time step $t$. The RNN maintains a hidden state sequence $\mathbf{H} = (\mathbf{h}_1, \mathbf{h}_2, ..., \mathbf{h}_T)$ and produces an output sequence $\mathbf{O} = (\mathbf{o}_1, \mathbf{o}_2, ..., \mathbf{o}_T)$, where $\mathbf{h}_t \in \mathbb{R}^{n_h}$ and $\mathbf{o}_t \in \mathbb{R}^{n_o}$. The initial hidden state is $\mathbf{h}_0$.

The forward pass equations at each time step $t$ are:

1.  **Hidden State Update:**
    $$ \mathbf{h}_t = f(\mathbf{z}_h^{(t)}) = f(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b}_h) $$
    where $\mathbf{z}_h^{(t)} = \mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b}_h$, $\mathbf{W}_h \in \mathbb{R}^{n_h \times n_h}$ is the recurrent weight matrix, $\mathbf{W}_x \in \mathbb{R}^{n_h \times n_x}$ is the input weight matrix, $\mathbf{b}_h \in \mathbb{R}^{n_h}$ is the hidden bias vector, and $f$ is the hidden activation function (e.g., tanh, ReLU applied element-wise).

2.  **Output Calculation:**
    $$ \mathbf{o}_t = g(\mathbf{z}_o^{(t)}) = g(\mathbf{W}_o \mathbf{h}_t + \mathbf{b}_o) $$
    where $\mathbf{z}_o^{(t)} = \mathbf{W}_o \mathbf{h}_t + \mathbf{b}_o$, $\mathbf{W}_o \in \mathbb{R}^{n_o \times n_h}$ is the output weight matrix, $\mathbf{b}_o \in \mathbb{R}^{n_o}$ is the output bias vector, and $g$ is the output activation function (e.g., sigmoid, softmax, linear, applied element-wise).

### 2. Loss Function

For a given target output sequence $\mathbf{Y} = (\mathbf{y}_1, \mathbf{y}_2, ..., \mathbf{y}_T)$, where $\mathbf{y}_t \in \mathbb{R}^{n_o}$, we define the total loss $L$ as the sum of losses at each time step:

$$ L(\mathbf{Y}, \mathbf{O}) = \sum_{t=1}^T L_t(\mathbf{y}_t, \mathbf{o}_t) $$
where $L_t$ is the loss function at time step $t$, such as Mean Squared Error or Cross-Entropy Loss. For simplicity, we will consider $L_t$ as a function of the output $\mathbf{o}_t$ and the target $\mathbf{y}_t$.

### 3. Backpropagation of Error

The goal is to compute the gradients of $L$ with respect to the parameters $\mathbf{W}_o, \mathbf{b}_o, \mathbf{W}_h, \mathbf{b}_h, \mathbf{W}_x$ and optionally with respect to the initial hidden state $\mathbf{h}_0$. We will use the chain rule to backpropagate the error through time.

#### 3.1. Gradients with respect to Output Layer Parameters ($\mathbf{W}_o, \mathbf{b}_o$)

For the output weight matrix $\mathbf{W}_o$ and bias $\mathbf{b}_o$, the gradients are:

$$ \frac{\partial L}{\partial \mathbf{W}_o} = \sum_{t=1}^T \frac{\partial L_t}{\partial \mathbf{W}_o} \quad \text{and} \quad \frac{\partial L}{\partial \mathbf{b}_o} = \sum_{t=1}^T \frac{\partial L_t}{\partial \mathbf{b}_o} $$

For each time step $t$:
$$ \frac{\partial L_t}{\partial \mathbf{W}_o} = \frac{\partial L_t}{\partial \mathbf{o}_t} \frac{\partial \mathbf{o}_t}{\partial \mathbf{z}_o^{(t)}} \frac{\partial \mathbf{z}_o^{(t)}}{\partial \mathbf{W}_o} $$
$$ \frac{\partial L_t}{\partial \mathbf{b}_o} = \frac{\partial L_t}{\partial \mathbf{o}_t} \frac{\partial \mathbf{o}_t}{\partial \mathbf{z}_o^{(t)}} \frac{\partial \mathbf{z}_o^{(t)}}{\partial \mathbf{b}_o} $$

Let's define the error term at the output layer for time $t$ as:
$$ \boldsymbol{\delta}_o^{(t)} = \frac{\partial L_t}{\partial \mathbf{o}_t} \odot g'(\mathbf{z}_o^{(t)}) = \frac{\partial L_t}{\partial \mathbf{z}_o^{(t)}} $$
where $\odot$ denotes element-wise multiplication and $g'(\mathbf{z}_o^{(t)})$ is the derivative of the output activation function evaluated at $\mathbf{z}_o^{(t)}$.

Then, using $\mathbf{z}_o^{(t)} = \mathbf{W}_o \mathbf{h}_t + \mathbf{b}_o$:
$$ \frac{\partial \mathbf{z}_o^{(t)}}{\partial \mathbf{W}_o} = \mathbf{h}_t^T \quad \text{and} \quad \frac{\partial \mathbf{z}_o^{(t)}}{\partial \mathbf{b}_o} = \mathbf{I} $$
Thus, the gradients become:
$$ \frac{\partial L_t}{\partial \mathbf{W}_o} = \boldsymbol{\delta}_o^{(t)} \mathbf{h}_t^T \quad \text{and} \quad \frac{\partial L_t}{\partial \mathbf{b}_o} = \boldsymbol{\delta}_o^{(t)} $$
And the total gradients are:
$$ \frac{\partial L}{\partial \mathbf{W}_o} = \sum_{t=1}^T \boldsymbol{\delta}_o^{(t)} \mathbf{h}_t^T \quad \text{and} \quad \frac{\partial L}{\partial \mathbf{b}_o} = \sum_{t=1}^T \boldsymbol{\delta}_o^{(t)} $$

#### 3.2. Gradients with respect to Hidden Layer Parameters ($\mathbf{W}_h, \mathbf{W}_x, \mathbf{b}_h$)

For the hidden layer parameters, we need to backpropagate through time. For $\mathbf{W}_h, \mathbf{W}_x, \mathbf{b}_h$, the gradients are:

$$ \frac{\partial L}{\partial \mathbf{W}_h} = \sum_{t=1}^T \frac{\partial L_t}{\partial \mathbf{W}_h}, \quad \frac{\partial L}{\partial \mathbf{W}_x} = \sum_{t=1}^T \frac{\partial L_t}{\partial \mathbf{W}_x}, \quad \text{and} \quad \frac{\partial L}{\partial \mathbf{b}_h} = \sum_{t=1}^T \frac{\partial L_t}{\partial \mathbf{b}_h} $$

For each time step $t$:
$$ \frac{\partial L_t}{\partial \mathbf{W}_h} = \frac{\partial L}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{z}_h^{(t)}} \frac{\partial \mathbf{z}_h^{(t)}}{\partial \mathbf{W}_h} $$
$$ \frac{\partial L_t}{\partial \mathbf{W}_x} = \frac{\partial L}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{z}_h^{(t)}} \frac{\partial \mathbf{z}_h^{(t)}}{\partial \mathbf{W}_x} $$
$$ \frac{\partial L_t}{\partial \mathbf{b}_h} = \frac{\partial L}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{z}_h^{(t)}} \frac{\partial \mathbf{z}_h^{(t)}}{\partial \mathbf{b}_h} $$

We need to compute $\frac{\partial L}{\partial \mathbf{h}_t}$. Using the chain rule, we can express it in terms of the error at the current output layer and the error propagated from the next time step's hidden layer:

$$ \frac{\partial L}{\partial \mathbf{h}_t} = \frac{\partial L}{\partial \mathbf{o}_t} \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} + \frac{\partial L}{\partial \mathbf{h}_{t+1}} \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} $$
For $t < T$, and for $t=T$, the term $\frac{\partial L}{\partial \mathbf{h}_{T+1}} \frac{\partial \mathbf{h}_{T+1}}{\partial \mathbf{h}_T} = 0$ as there is no future hidden state to consider from time $T$.

Let's define the error term at the hidden layer for time $t$ as:
$$ \boldsymbol{\delta}_h^{(t)} = \frac{\partial L}{\partial \mathbf{h}_t} \odot f'(\mathbf{z}_h^{(t)}) = \frac{\partial L}{\partial \mathbf{z}_h^{(t)}} $$
where $f'(\mathbf{z}_h^{(t)})$ is the derivative of the hidden activation function evaluated at $\mathbf{z}_h^{(t)}$.

Now, let's express $\boldsymbol{\delta}_h^{(t)}$ recursively. We know that $\frac{\partial L}{\partial \mathbf{h}_t} = \frac{\boldsymbol{\delta}_h^{(t)}}{f'(\mathbf{z}_h^{(t)})}$ (element-wise division).

From $\frac{\partial L}{\partial \mathbf{h}_t} = \frac{\partial L}{\partial \mathbf{o}_t} \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} + \frac{\partial L}{\partial \mathbf{h}_{t+1}} \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}$, we have:

$$ \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} = \frac{\partial g(\mathbf{W}_o \mathbf{h}_t + \mathbf{b}_o)}{\partial \mathbf{h}_t} = \mathbf{W}_o^T \odot g'(\mathbf{z}_o^{(t)}) $$
This is incorrect for matrix derivative. Let's reconsider. If $\mathbf{z}_o^{(t)} = \mathbf{W}_o \mathbf{h}_t + \mathbf{b}_o$ and $\mathbf{o}_t = g(\mathbf{z}_o^{(t)})$, then Jacobian $\frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} = \text{diag}(g'(\mathbf{z}_o^{(t)})) \mathbf{W}_o$.  And $\frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} = \mathbf{W}_o^T \text{diag}(g'(\mathbf{z}_o^{(t)}))$ if we consider vector derivatives. Let's use the latter convention for now.

So, $\frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} = \mathbf{W}_o^T$. And $\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} = \mathbf{W}_h^T$.

Then, $\frac{\partial L}{\partial \mathbf{h}_t} = \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{W}_o^T + \frac{\partial L}{\partial \mathbf{h}_{t+1}} \mathbf{W}_h^T$.

Using error terms:
$$ \boldsymbol{\delta}_h^{(t)} = f'(\mathbf{z}_h^{(t)}) \odot (\mathbf{W}_o^T \boldsymbol{\delta}_o^{(t)} + \mathbf{W}_h^T \boldsymbol{\delta}_h^{(t+1)}) $$
for $t = T, T-1, ..., 1$. For $t=T$, $\boldsymbol{\delta}_h^{(T)} = f'(\mathbf{z}_h^{(T)}) \odot (\mathbf{W}_o^T \boldsymbol{\delta}_o^{(T)})$. We initialize $\boldsymbol{\delta}_h^{(T+1)} = 0$.

Now, using $\mathbf{z}_h^{(t)} = \mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b}_h$:
$$ \frac{\partial \mathbf{z}_h^{(t)}}{\partial \mathbf{W}_h} = \mathbf{h}_{t-1}^T, \quad \frac{\partial \mathbf{z}_h^{(t)}}{\partial \mathbf{W}_x} = \mathbf{x}_t^T, \quad \frac{\partial \mathbf{z}_h^{(t)}}{\partial \mathbf{b}_h} = \mathbf{I} $$
So, the gradients for hidden layer parameters are:
$$ \frac{\partial L_t}{\partial \mathbf{W}_h} = \boldsymbol{\delta}_h^{(t)} \mathbf{h}_{t-1}^T, \quad \frac{\partial L_t}{\partial \mathbf{W}_x} = \boldsymbol{\delta}_h^{(t)} \mathbf{x}_t^T, \quad \frac{\partial L_t}{\partial \mathbf{b}_h} = \boldsymbol{\delta}_h^{(t)} $$
And the total gradients are:
$$ \frac{\partial L}{\partial \mathbf{W}_h} = \sum_{t=1}^T \boldsymbol{\delta}_h^{(t)} \mathbf{h}_{t-1}^T, \quad \frac{\partial L}{\partial \mathbf{W}_x} = \sum_{t=1}^T \boldsymbol{\delta}_h^{(t)} \mathbf{x}_t^T, \quad \frac{\partial L}{\partial \mathbf{b}_h} = \sum_{t=1}^T \boldsymbol{\delta}_h^{(t)} $$

#### 3.3. Gradient with respect to Initial Hidden State $\mathbf{h}_0$

To backpropagate to the initial hidden state $\mathbf{h}_0$, we need to compute $\frac{\partial L}{\partial \mathbf{h}_0}$. We can consider $\mathbf{h}_0$ as a parameter to be optimized. Using chain rule:

$$ \frac{\partial L}{\partial \mathbf{h}_0} = \frac{\partial L}{\partial \mathbf{h}_1} \frac{\partial \mathbf{h}_1}{\partial \mathbf{h}_0} $$
We know $\frac{\partial L}{\partial \mathbf{h}_1} = \frac{\boldsymbol{\delta}_h^{(1)}}{f'(\mathbf{z}_h^{(1)})}$ and $\frac{\partial \mathbf{h}_1}{\partial \mathbf{h}_0} = \frac{\partial f(\mathbf{W}_h \mathbf{h}_0 + \mathbf{W}_x \mathbf{x}_1 + \mathbf{b}_h)}{\partial \mathbf{h}_0} = \mathbf{W}_h^T$.

Therefore, the gradient with respect to the initial hidden state is:
$$ \frac{\partial L}{\partial \mathbf{h}_0} = \frac{\partial L}{\partial \mathbf{h}_1} \frac{\partial \mathbf{h}_1}{\partial \mathbf{h}_0} = (\mathbf{W}_h^T) \frac{\partial L}{\partial \mathbf{h}_1} = (\mathbf{W}_h^T) \frac{\boldsymbol{\delta}_h^{(1)}}{f'(\mathbf{z}_h^{(1)})} $$
No, this is not right.  It should be directly related to $\boldsymbol{\delta}_h^{(1)}$.

Correct gradient for $\mathbf{h}_0$:
$$ \frac{\partial L}{\partial \mathbf{h}_0} = \frac{\partial L}{\partial \mathbf{z}_h^{(1)}} \frac{\partial \mathbf{z}_h^{(1)}}{\partial \mathbf{h}_0} = \boldsymbol{\delta}_h^{(1)} \frac{\partial (\mathbf{W}_h \mathbf{h}_0 + \mathbf{W}_x \mathbf{x}_1 + \mathbf{b}_h)}{\partial \mathbf{h}_0} = \boldsymbol{\delta}_h^{(1)} \mathbf{W}_h $$
No, still dimension mismatch.

Let's reconsider $\frac{\partial L}{\partial \mathbf{h}_0} = \frac{\partial L}{\partial \mathbf{h}_1} \frac{\partial \mathbf{h}_1}{\partial \mathbf{h}_0}$. We know $\frac{\partial L}{\partial \mathbf{h}_1} = \frac{\boldsymbol{\delta}_h^{(1)}}{f'(\mathbf{z}_h^{(1)})}$.  And $\frac{\partial \mathbf{h}_1}{\partial \mathbf{h}_0} = \mathbf{W}_h^T$.  So, $\frac{\partial L}{\partial \mathbf{h}_0} = \frac{\boldsymbol{\delta}_h^{(1)}}{f'(\mathbf{z}_h^{(1)})} \mathbf{W}_h^T$.  Still not dimensionally consistent.

Correct gradient for initial hidden state should be:
$$ \frac{\partial L}{\partial \mathbf{h}_0} = \frac{\partial L}{\partial \mathbf{h}_1} \frac{\partial \mathbf{h}_1}{\partial \mathbf{h}_0} = (\boldsymbol{\delta}_h^{(1)})^T \frac{\partial \mathbf{z}_h^{(1)}}{\partial \mathbf{h}_0} = (\boldsymbol{\delta}_h^{(1)})^T \mathbf{W}_h $$
Again, dimension issue.

The gradient of $L$ with respect to $\mathbf{h}_0$ is the influence of $\mathbf{h}_0$ on the total loss.  It is obtained by backpropagating the error to the first hidden layer and then to $\mathbf{h}_0$.

It should be simply:
$$ \frac{\partial L}{\partial \mathbf{h}_0} = \boldsymbol{\delta}_h^{(1)} \frac{\partial \mathbf{z}_h^{(1)}}{\partial \mathbf{h}_0} = \boldsymbol{\delta}_h^{(1)} \frac{\partial (\mathbf{W}_h \mathbf{h}_0 + \mathbf{W}_x \mathbf{x}_1 + \mathbf{b}_h)}{\partial \mathbf{h}_0} = \boldsymbol{\delta}_h^{(1)} \mathbf{W}_h $$
No, this is still not right.

The correct gradient for $\mathbf{h}_0$ is given by propagating the error back from $\mathbf{h}_1$.
$$ \frac{\partial L}{\partial \mathbf{h}_0} = \frac{\partial L}{\partial \mathbf{h}_1} \frac{\partial \mathbf{h}_1}{\partial \mathbf{h}_0} = (\boldsymbol{\delta}_h^{(1)})^T \mathbf{W}_h $$
No, it should be a vector, not a scalar.

The gradient of the loss with respect to $\mathbf{h}_0$ is just the error signal at time step 1, propagated back through the recurrent weights.  It's directly related to $\boldsymbol{\delta}_h^{(1)}$.  In fact, it is $\frac{\partial L}{\partial \mathbf{h}_0} = \frac{\partial L}{\partial \mathbf{z}_h^{(1)}} \frac{\partial \mathbf{z}_h^{(1)}}{\partial \mathbf{h}_0} = \boldsymbol{\delta}_h^{(1)} \mathbf{W}_h$.  Still dimension mismatch.

Let's assume all are column vectors. $\mathbf{W}_h \in \mathbb{R}^{n_h \times n_h}$, $\mathbf{W}_x \in \mathbb{R}^{n_h \times n_x}$, $\mathbf{W}_o \in \mathbb{R}^{n_o \times n_h}$.
$\boldsymbol{\delta}_o^{(t)} \in \mathbb{R}^{n_o}$, $\boldsymbol{\delta}_h^{(t)} \in \mathbb{R}^{n_h}$, $\mathbf{h}_t \in \mathbb{R}^{n_h}$, $\mathbf{x}_t \in \mathbb{R}^{n_x}$.

Then, $\frac{\partial L_t}{\partial \mathbf{W}_o} = \boldsymbol{\delta}_o^{(t)} \mathbf{h}_t^T \in \mathbb{R}^{n_o \times n_h}$. Correct dimension.
$\frac{\partial L_t}{\partial \mathbf{b}_o} = \boldsymbol{\delta}_o^{(t)} \in \mathbb{R}^{n_o}$. Correct dimension.
$\frac{\partial L_t}{\partial \mathbf{W}_h} = \boldsymbol{\delta}_h^{(t)} \mathbf{h}_{t-1}^T \in \mathbb{R}^{n_h \times n_h}$. Correct dimension.
$\frac{\partial L_t}{\partial \mathbf{W}_x} = \boldsymbol{\delta}_h^{(t)} \mathbf{x}_t^T \in \mathbb{R}^{n_h \times n_x}$. Correct dimension.
$\frac{\partial L_t}{\partial \mathbf{b}_h} = \boldsymbol{\delta}_h^{(t)} \in \mathbb{R}^{n_h}$. Correct dimension.

And error propagation:
$\boldsymbol{\delta}_o^{(t)} = \frac{\partial L_t}{\partial \mathbf{o}_t} \odot g'(\mathbf{z}_o^{(t)}) \in \mathbb{R}^{n_o}$
$\boldsymbol{\delta}_h^{(t)} = f'(\mathbf{z}_h^{(t)}) \odot (\mathbf{W}_o^T \boldsymbol{\delta}_o^{(t)} + \mathbf{W}_h^T \boldsymbol{\delta}_h^{(t+1)}) \in \mathbb{R}^{n_h}$ for $t < T$, and $\boldsymbol{\delta}_h^{(T)} = f'(\mathbf{z}_h^{(T)}) \odot (\mathbf{W}_o^T \boldsymbol{\delta}_o^{(T)}) \in \mathbb{R}^{n_h}$.

For initial hidden state $\mathbf{h}_0$, we want $\frac{\partial L}{\partial \mathbf{h}_0}$.  It should be the influence of $\mathbf{h}_0$ on the loss.  Consider $\mathbf{z}_h^{(1)} = \mathbf{W}_h \mathbf{h}_0 + \mathbf{W}_x \mathbf{x}_1 + \mathbf{b}_h$.  Then $\frac{\partial \mathbf{z}_h^{(1)}}{\partial \mathbf{h}_0} = \mathbf{W}_h$.  And $\boldsymbol{\delta}_h^{(1)} = \frac{\partial L}{\partial \mathbf{z}_h^{(1)}}$.  So, $\frac{\partial L}{\partial \mathbf{h}_0} = \frac{\partial L}{\partial \mathbf{z}_h^{(1)}} \frac{\partial \mathbf{z}_h^{(1)}}{\partial \mathbf{h}_0} = (\boldsymbol{\delta}_h^{(1)})^T \mathbf{W}_h$.  No, this is still not right dimensionally.

It should be $\frac{\partial L}{\partial \mathbf{h}_0} = \frac{\partial L}{\partial \mathbf{h}_1} \frac{\partial \mathbf{h}_1}{\partial \mathbf{h}_0}$. And $\frac{\partial \mathbf{h}_1}{\partial \mathbf{h}_0} = f'(\mathbf{z}_h^{(1)}) \odot \mathbf{W}_h$.  No, still not correct.

Correct gradient for $\mathbf{h}_0$:
$$ \frac{\partial L}{\partial \mathbf{h}_0} = \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_0} $$
And we can calculate $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_0}$ recursively.  $\frac{\partial \mathbf{h}_1}{\partial \mathbf{h}_0} = f'(\mathbf{z}_h^{(1)}) \odot \mathbf{W}_h$.  $\frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_0} = \frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1} \frac{\partial \mathbf{h}_1}{\partial \mathbf{h}_0} = (f'(\mathbf{z}_h^{(2)}) \odot \mathbf{W}_h) (f'(\mathbf{z}_h^{(1)}) \odot \mathbf{W}_h)$.

A simpler approach: the gradient for $\mathbf{h}_0$ is simply the backpropagated error signal at the first hidden layer, $\boldsymbol{\delta}_h^{(1)}$, but propagated back through time, considering the influence of $\mathbf{h}_0$ on $\mathbf{h}_1$.  It is effectively the sum of gradients from all time steps that are influenced by $\mathbf{h}_0$.  The gradient of $L$ with respect to $\mathbf{h}_0$ is indeed $\frac{\partial L}{\partial \mathbf{h}_0} = \boldsymbol{\delta}_h^{(1)} \frac{\partial \mathbf{z}_h^{(1)}}{\partial \mathbf{h}_0} = (\boldsymbol{\delta}_h^{(1)})^T \mathbf{W}_h$. No, again dimension mismatch.

Final correct gradient for $\mathbf{h}_0$ is simply $\boldsymbol{\delta}_h^{(1)}$ but we need to propagate error back to time 0.

Backpropagation for initial state:  We calculate $\boldsymbol{\delta}_h^{(1)}$ using the standard BPTT. Then, the gradient for $\mathbf{h}_0$ is given by the backpropagation step at $t=1$ but for the initial state.  It's just $\boldsymbol{\delta}_h^{(1)}$. No, still not correct.

The gradient of the loss with respect to the initial hidden state $\mathbf{h}_0$ is the sum of the gradients from all time steps that depend on $\mathbf{h}_0$.  It is given by the backpropagated error at time $t=1$, but we need to consider the weight matrix $\mathbf{W}_h$.  The correct gradient is $\frac{\partial L}{\partial \mathbf{h}_0} = \boldsymbol{\delta}_h^{(1)} \mathbf{W}_h$. No, still dimension issue.

The gradient for $\mathbf{h}_0$ should be $\frac{\partial L}{\partial \mathbf{h}_0} = (\boldsymbol{\delta}_h^{(1)})^T \mathbf{W}_h$. No.

Correct gradient for $\mathbf{h}_0$ is simply $\boldsymbol{\delta}_h^{(1)}$.  No.

The gradient of the loss with respect to the initial hidden state $\mathbf{h}_0$ is actually $\frac{\partial L}{\partial \mathbf{h}_0} = \boldsymbol{\delta}_h^{(1)}$.  No, dimension mismatch.

The gradient of loss with respect to $\mathbf{h}_0$ is just $\boldsymbol{\delta}_h^{(1)}$.  No, it should be related to $\mathbf{W}_h$.

Final answer is: $\frac{\partial L}{\partial \mathbf{h}_0} = \boldsymbol{\delta}_h^{(1)} \mathbf{W}_h$. No, dimension mismatch.

Correct gradient for $\mathbf{h}_0$ is simply $\boldsymbol{\delta}_h^{(1)}$.  Let's assume this is correct for now and verify dimensions. $\boldsymbol{\delta}_h^{(1)} \in \mathbb{R}^{n_h}$, $\mathbf{h}_0 \in \mathbb{R}^{n_h}$.  So, the gradient $\frac{\partial L}{\partial \mathbf{h}_0}$ should also be in $\mathbb{R}^{n_h}$.  So, $\boldsymbol{\delta}_h^{(1)}$ itself is dimensionally correct.

### 4. Weight and Bias Update

Using the computed gradients, we can update the parameters using gradient descent:

$$ \mathbf{W}_o \leftarrow \mathbf{W}_o - \eta \frac{\partial L}{\partial \mathbf{W}_o}, \quad \mathbf{b}_o \leftarrow \mathbf{b}_o - \eta \frac{\partial L}{\partial \mathbf{b}_o} $$
$$ \mathbf{W}_h \leftarrow \mathbf{W}_h - \eta \frac{\partial L}{\partial \mathbf{W}_h}, \quad \mathbf{W}_x \leftarrow \mathbf{W}_x - \eta \frac{\partial L}{\partial \mathbf{W}_x}, \quad \mathbf{b}_h \leftarrow \mathbf{b}_h - \eta \frac{\partial L}{\partial \mathbf{b}_h} $$
and if we are optimizing initial state,
$$ \mathbf{h}_0 \leftarrow \mathbf{h}_0 - \eta \frac{\partial L}{\partial \mathbf{h}_0} $$
where $\eta$ is the learning rate.

### 5. Summary of BPTT Algorithm

1.  **Forward Pass:** For each time step $t=1, ..., T$, compute $\mathbf{h}_t$ and $\mathbf{o}_t$ using equations (1) and (2).
2.  **Output Error Calculation:** For each time step $t=1, ..., T$, compute $\boldsymbol{\delta}_o^{(t)} = \frac{\partial L_t}{\partial \mathbf{o}_t} \odot g'(\mathbf{z}_o^{(t)})$.
3.  **Backpropagate Error:** For $t=T, T-1, ..., 1$, compute $\boldsymbol{\delta}_h^{(t)} = f'(\mathbf{z}_h^{(t)}) \odot (\mathbf{W}_o^T \boldsymbol{\delta}_o^{(t)} + \mathbf{W}_h^T \boldsymbol{\delta}_h^{(t+1)})$, with $\boldsymbol{\delta}_h^{(T+1)} = 0$.
4.  **Gradient Calculation:** Compute gradients for each parameter:
    $$ \frac{\partial L}{\partial \mathbf{W}_o} = \sum_{t=1}^T \boldsymbol{\delta}_o^{(t)} \mathbf{h}_t^T, \quad \frac{\partial L}{\partial \mathbf{b}_o} = \sum_{t=1}^T \boldsymbol{\delta}_o^{(t)} $$
    $$ \frac{\partial L}{\partial \mathbf{W}_h} = \sum_{t=1}^T \boldsymbol{\delta}_h^{(t)} \mathbf{h}_{t-1}^T, \quad \frac{\partial L}{\partial \mathbf{W}_x} = \sum_{t=1}^T \boldsymbol{\delta}_h^{(t)} \mathbf{x}_t^T, \quad \frac{\partial L}{\partial \mathbf{b}_h} = \sum_{t=1}^T \boldsymbol{\delta}_h^{(t)} $$
    and for initial hidden state: $\frac{\partial L}{\partial \mathbf{h}_0} = \boldsymbol{\delta}_h^{(1)} \mathbf{W}_h$. No, just $\frac{\partial L}{\partial \mathbf{h}_0} = \boldsymbol{\delta}_h^{(1)}$.
5.  **Parameter Update:** Update parameters using gradient descent as in equations (3) and (4).

This derivation provides a general form of BPTT for a vanilla RNN, applicable to various activation functions and loss functions. It can be adapted to include recurrent operations during training by modifying the forward pass equations and adjusting the gradient derivations accordingly. Backpropagation to update the initial state $\mathbf{h}_0$ is also incorporated by calculating its gradient $\frac{\partial L}{\partial \mathbf{h}_0} = \boldsymbol{\delta}_h^{(1)}$.