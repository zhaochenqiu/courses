import matplotlib.pyplot as plt
import pickle

with open('Matrix_layer.pkl', 'rb') as f:
    result_errors1 = pickle.load(f)

with open('Matrix.pkl', 'rb') as f:
    result_errors2 = pickle.load(f)





plt.figure()
plt.plot(range(len(result_errors1)), result_errors1, 'r', range(len(result_errors2)), result_errors2, 'b' )
plt.show()



# plt.figure()
# plt.plot(range(len(result_errors)), result_errors)
#
# plt.show()


