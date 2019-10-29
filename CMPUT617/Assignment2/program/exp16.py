import torch

x = torch.randn(128, 20)  # 输入的维度是（128，20）
m = torch.nn.Linear(20, 30)  # 20,30是指维度
output = m(x)
print('m.weight.shape:\n ', m.weight.shape)
print('m.bias.shape:\n', m.bias.shape)
print('output.shape:\n', output.shape)


A = torch.randn(4, 4)*10

y = A*A

print("A:", A)
print("y:", y)



A = torch.randn(1, 4)*10

y = A*A

print("A:", A)
print("y:", y)






# ans = torch.mm(input,torch.t(m.weight))+m.bias 等价于下面的
ans = torch.mm(x, m.weight.t()) + m.bias
print('ans.shape:\n', ans.shape)

print(torch.equal(ans, output))
