import torch

torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

# version 1
# xbow = torch.zeros((B, T, C))
# for b in range(B):
#   for t in range(T):
#       xprev = x[b, :t + 1] # shape: (t, C)
#       xbow[b, t] = torch.mean(xprev, dim=0)  # shape: (C,)

# version 2
# w = torch.tril(torch.ones(T, T))
# w /= w.sum(1, keepdim=True)
# xbow = w @ x
# print(x[0])
# print(xbow[0])

# version 3
# tril = torch.tril(torch.ones(T, T))
# w = torch.zeros((T, T))
# w = w.masked_fill(tril == 0, float("-inf"))  # Can't see the future
# w = torch.nn.functional.softmax(w, dim=-1)  # Trick that makes -inf -> zero and non inf get normalized to sum to 1
# xbow = w @ x
# print(x[0])
# print(xbow[0])

# version 4: self attention
head_size = 16
query = torch.nn.Linear(C, head_size, bias=False)
key = torch.nn.Linear(C, head_size, bias=False)
value = torch.nn.Linear(C, head_size, bias=False)
q = query(x)  # (B, T, H)
k = key(x)  # (B, T, H)
w = q @ k.transpose(-2, -1)  # (B, T, H) @ (B, H, T) -> (B, T, T)

"""divide by sqrt headsize makes weights more evenly distributed. Takes variance from ~head_size to ~1.0.
Without this, softmax will make attention go to maybe 1 other token instead of being diffuse
"""
w = q @ k.transpose(-2, -1) * head_size ** -0.5

tril = torch.tril(torch.ones(T, T))
# w = torch.zeros((T, T))
w = w.masked_fill(tril == 0, float("-inf"))  # Can't see the future. Don't use this if you're not predicting the future
w = torch.nn.functional.softmax(w, dim=-1)  # Trick that makes -inf -> zero and non inf get normalized to sum to 1

v = value(x)  # What gets aggregated between heads (dot product with attention weights)
xbow = w @ v
# xbow = w @ x
print(w[0])
print(x[0])
print(xbow[0])