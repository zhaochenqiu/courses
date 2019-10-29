import torch

x = torch.randn(6, 3)  # 输入的维度是（128，20）
m = torch.nn.Linear(3, 4)  # 20,30是指维度
output = m(x)
print('m.weight.shape:\n ', m.weight.shape)
print('m.bias.shape:\n', m.bias.shape)
print('output.shape:\n', output.shape)




# ans = torch.mm(input,torch.t(m.weight))+m.bias 等价于下面的

temp1 = torch.mm(x, m.weight.t())
print("x:", x)
print("m.weight:", m.weight.t())
print("temp1:", temp1)


ans = temp1 + m.bias

print("\n\n\n")
print("temp1:", temp1)
print("\n")
print("m.bias:", m.bias)
print("\n")
print("ans:", ans)


# ans = torch.mm(x, m.weight.t()) + m.bias
print('ans.shape:\n', ans.shape)

print(torch.equal(ans, output))
