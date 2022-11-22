# you are not allowed to import other package
import torch

def question1(shape):
    """
    Define a function to return a tensor with all elements are number one with a given shape
    """
    
    return torch.ones(shape)


def question2(data):
    """
    Define a function to convert a python list into a pytorch tensor with torch.long data type
    """
    
    return torch.Tensor(data).long()


def question3(a, b):
    """
    Define a function to compute 3*a+2*b
    """
    assert a.size() == b.size() , "Input tensors should have the same shape"
    return 3 * a + 2 * b


def question4(a):
    """
    Define a function to get the last column from a 2 dimensional tensor
    """
    assert a.dim() == 2 , "Tensor should be 2-dimensional"
    return a[:, -1]


def question5(data):
    """
    Define a function to combine a list of tensors into a new tensor at the last dimension, then expand the 1st dimension
    """
    
    return torch.stack([torch.stack(data, dim=-1)], dim=0)


def question6(data):
    """
    Define a function to combine a list of 1-D tensors with different lengths into a new tensor by padding the shorter tensors with 0 on the left side
    """
    # get max size
    max_size = 0
    for tensor in data:
        assert tensor.dim() == 1 , "Tensor must be 1-dimensional"
        curr_size = tensor.size(dim=0)
        if curr_size > max_size:
            max_size = curr_size
    
    # pad on the left so new tensor size = max size
    data = [torch.nn.functional.pad(tensor, (max_size-tensor.size(dim=0), 0), value=0) for tensor in data]
    return torch.stack(data)


def question7(y, w, b):
    """
    Define a function that calculates w*(y - b)
    """

    return w.T @ (y - b)


def question8(y, w, b):
    """
    Define a function that calculates batch w*(y - b)
    DO NOT use loop, list comprehension, or any other similar operations.
    """
    
    return torch.bmm(torch.transpose(w, 1, 2), y - b)


def question9(x):
    """
    Given a 3-D tensor x (b, n, m), calculate the mean along the n dimension without accounting for the 0-values.
    DO NOT use loop, list comprehension, or any other similar operations.
    """
    mask = x != 0
    return x.sum(dim=-1) / mask.sum(dim=-1)

# q10 helper
def euclidean_dist(a, b):
    a = torch.Tensor(a)
    b = torch.Tensor(b)
    square_dist = ((a-b)**2).sum()
    return square_dist**0.5

def question10(pairs):
    """
    Define a funtion that calculates the Euclidean distance of each vector pair.
    """
    result = [euclidean_dist(a, b) for a, b in pairs]
    return torch.stack(result, dim=0)

#TODO: fix the examples
def main():
    q1_input = (2,3)
    print('Q1 Example input: {}\n'.format(q1_input))
    q1 = question1(q1_input)
    print('Q1 example output: \n{}\n'.format(q1))
    q2_input = [[1., 2.1, 3.0], [4., 5., 6.2]]
    print('Q2 Example input: \n{}\n'.format(q2_input))
    q2 = question2(q2_input)
    print('Q2 example output: \n{}\n'.format(q2))
    print('Q3 Example input: \na: {}\nb: {}\n'.format(q2, question2([[1,1,1], [1,1,1]])))
    q3 = question3(q2, question2([[1,1,1], [1,1,1]]))
    print('Q3 example output: \n{}\n'.format(q3))
    print('Q4 Example input: \n{}\n'.format(q2))
    print('Q4 example output: \n{}\n'.format(question4(q2)))
    q5_input = [question4(q1).type(torch.long), question4(q2), question4(q3)]
    print('Q5 Example input: \n{}\n'.format(q5_input))
    q5 = question5(q5_input)
    print('Q5 example output: \n{}\n'.format(q5))
    q6_input = [question2([1]), question2([2, 2]), question2([3, 3, 3])]
    print('Q6 Example input: \n{}\n'.format(q6_input))
    q6 = question6(q6_input)
    print('Q6 example output: \n{}\n'.format(q6))
    q7_input = (torch.tensor([[1.12, 1.57, 2.11], [0.23, 0.72, 1.19], [0.52, 0.4, 0.31]]),
                torch.tensor([[0.3, 0.2], [0.6, -0.1], [-0.3, 0.2]]),
                torch.tensor([[0.02, -0.03, 0.01], [0.03, 0.02, -0.01], [0.02, 0, 0.01]]))
    print('Q7 Example input \ny: \n{}\nw: {}\nb: {}\n'.format(*q7_input))
    q7 = question7(*q7_input)
    print('Q7 example output: \n{}\n'.format(q7))
    q8_input = (torch.tensor([[[1.12, 1.57, 2.11], [0.23, 0.72, 1.19], [0.52, 0.4, 0.31]], [[0.31, 0.36, 0.3], [0.82, 0.51, 0.78], [0.21, -0.01, 0.22]]]),
                torch.tensor([[[0.3, 0.2], [0.6, -0.1], [-0.3, 0.2]], [[-0.6, 0.5], [0.1, 0.2], [0.4, -0.2]]]),
                torch.tensor([[[0.02, -0.03, 0.01], [0.03, 0.02, -0.01], [0.02, 0, 0.01]],[[0.01, -0.04, 0.0], [0.02, 0.01, -0.02], [0.01, -0.01, 0.02]]]))
    print('Q8 Example input \ny: \n{}\nw: {}\nb: {}\n'.format(*q8_input))
    q8 = question8(*q8_input)
    print('Q8 example output: \n{}\n'.format(q8))
    q9_input = torch.tensor([[[1.0, 0., 0.], [1.2, 0., 0.]], [[2.0, 2.2, 0.], [2.2, 2.6, 0.]], [[3.0, 3.2, 3.1], [3.2, 3.4, 3.6]]])
    print('Q9 Example input: \n{}\n'.format(q9_input))
    q9 = question9(q9_input)
    print('Q9 example output: \n{}\n'.format(q9))
    q10_input = [([1, 1, 1], [2, 2, 2]), ([1, 2, 3], [3, 2, 1]), ([0.1, 0.2, 0.3], [0.33, 0.25, 0.1])]
    print('Q10 Example input: \n{}\n'.format(q10_input))
    q10 = question10(q10_input)
    print('Q10 example output: \n{}\n'.format(q10))

    print('\n==== A2 Part 1 Done ====')


if __name__ == "__main__":
    main()