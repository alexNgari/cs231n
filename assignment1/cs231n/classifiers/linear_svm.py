from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,y[i]] -= X[i,:] 
                dW[:,j] += X[i,:] 
    

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW += 2*reg*W
    dW/=num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    N = X.shape[0] # X is NxD

    scores = X.dot(W)
    yi_scores = scores[np.arange(N), y]
    margins = np.maximum(0, scores - np.matrix(yi_scores).T +1)
    margins[np.arange(N), y] = 0
    loss = np.mean(np.sum(margins, axis=1)) + reg*np.sum(W*W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    mask = np.where(margins>0, 1, 0)                    # Makes a binary mask for where identity fn is 1 and j /= yi
    row_sum = np.sum(mask, axis=1)                      # Count all the non-zero classes for each X
    mask[np.arange(np.shape(X)[0]), y] = -1*row_sum.T   # To factor in the gradient wrt yi
    dW = (X.T).dot(mask)
    dW /= np.shape(X)[0]
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


# from builtins import range
# import numpy as np
# from random import shuffle
# from past.builtins import xrange

# def svm_loss_naive(W, X, y, reg):
#     """
#     Structured SVM loss function, naive implementation (with loops).

#     Inputs have dimension D, there are C classes, and we operate on minibatches
#     of N examples.

#     Inputs:
#     - W: A numpy array of shape (D, C) containing weights.
#     - X: A numpy array of shape (N, D) containing a minibatch of data.
#     - y: A numpy array of shape (N,) containing training labels; y[i] = c means
#       that X[i] has label c, where 0 <= c < C.
#     - reg: (float) regularization strength

#     Returns a tuple of:
#     - loss as single float
#     - gradient with respect to weights W; an array of same shape as W
#     """
#     dW = np.zeros(W.shape) # initialize the gradient as zero

#     # compute the loss and the gradient
#     num_dim, num_classes = W.shape
#     num_train = X.shape[0]
#     score_loss = 0.0
#     dWn = np.zeros((num_train, num_dim, num_classes))
#     for i in range(num_train):
#         scores = X[i].dot(W)
#         correct_class_score = scores[y[i]]
#         loss_classes = 0 # no. of classes that contribute to loss in this datapoint
#         for j in range(num_classes):
#             if j == y[i]:
#                 continue
#             margin = scores[j] - correct_class_score + 1 # note delta = 1
#             if margin > 0:
#                 loss_classes += 1
#                 # dW for other rows of W = Xi if contributes to loss, [0..] ow
#                 dWn[i, :, j] = X[i] 
#                 score_loss += margin
#         # dW for W row corresponding to correct class = -Xi*loss_classes
#         dWn[i, :, y[i]] = -loss_classes*X[i]

#     # Right now the loss is a sum over all training examples, but we want it
#     # to be an average instead so we divide by num_train.
#     score_loss /= num_train

#     # Add regularization to the loss.
#     reg_loss = reg * np.sum(W * W)

#     #############################################################################
#     # TODO:                                                                     #
#     # Compute the gradient of the loss function and store it dW.                #
#     # Rather that first computing the loss and then computing the derivative,   #
#     # it may be simpler to compute the derivative at the same time that the     #
#     # loss is being computed. As a result you may need to modify some of the    #
#     # code above to compute the gradient.                                       #
#     #############################################################################
#     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     dW_reg = 2*reg*W
#     # dW from scores is avg over N datapoints. Since dL/dLi = 1/N
#     dW = np.average(dWn, axis=0)

#     dW += dW_reg
#     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
#     return score_loss + reg_loss, dW



# def svm_loss_vectorized(W, X, y, reg):
#     """
#     Structured SVM loss function, vectorized implementation.

#     Inputs and outputs are the same as svm_loss_naive.
#     """
#     loss = 0.0
#     dW = np.zeros(W.shape) # initialize the gradient as zero

#     #############################################################################
#     # TODO:                                                                     #
#     # Implement a vectorized version of the structured SVM loss, storing the    #
#     # result in loss.                                                           #
#     #############################################################################
#     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     # y: A numpy array of shape (N,)

#     D, C = W.shape # W is DxC
#     N = X.shape[0] # X is NxD

#     scores = X.dot(W)                                       #     scores = X.dot(W)
#     # Sj-Syi+1
#     x = [i for i in range(N)]                               
#     scores = scores - scores[x,y].reshape(-1,1) + 1         #     yi_scores = scores[np.arange(np.shape(scores)[0]), y]
#     scores[x,y] = 0                                         #     margins = np.maximum(0, scores - np.matrix(yi_scores).T +1)
#     # Lij = min(0,Sj-Syi+1)
#     scores[scores < 0] = 0 
#     # L = 1/N*sum(sum(Lij))
#     loss = np.sum(scores) / N                               #     loss = np.mean(np.sum(margins, axis=1)) + 0.5*reg*np.sum(W*W)
#     # Add regularization to the loss.
#     loss += reg * np.sum(W * W)


#     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     #############################################################################
#     # TODO:                                                                     #
#     # Implement a vectorized version of the gradient for the structured SVM     #
#     # loss, storing the result in dW.                                           #
#     #                                                                           #
#     # Hint: Instead of computing the gradient from scratch, it may be easier    #
#     # to reuse some of the intermediate values that you used to compute the     #
#     # loss.                                                                     #
#     #############################################################################
#     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     # dW for other rows is Xi*(1 if is nonzero, 0 otherwise)
#     scores[scores > 0] = 1
#     # dW for rows corresponding to correct classes is -Xi*(count of nonzero classes)
#     nz = np.count_nonzero(scores, axis=1)
#     scores[x,y] = nz

#     # dW for all N has shape NxDxC. Broadcast to shape
#     dWn = scores.reshape(N,1,C) * X.reshape(N,D,1)

#     # dW is avg of dWn along N axis. Since dL/dLi = 1/L
#     dW = np.average(dWn, axis=0)

#     # add gradient from regulization path
#     dW += 2*reg*W

#     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     return loss, dW