using LinearAlgebra
function f(u)
    #the input vector u
    x, y, z, λ = u
    # Compute the system of equations
    eq1 = x^2 / 25 + y^2 / 16 + z^2 / 9 - 1
    eq2 = z - 3 - λ * z / 9
    eq3 = y - 4 - λ * y / 16
    eq4 = x - 5 - λ * x / 25
    # equations into a vector and return it
    return [eq1, eq2, eq3, eq4]
end
function find_jac(f, x, epsilon)
    # stores the length of vector field 
    n = length(x)         
    # create the n*n matrix and fills with zeroes                             
    J = zeros(n, n)  
      # iterate each column (4 times)                                  
    for j in 1:n        
        # create a vector of zeroes                             
        ei = zeros(n)              
        # set the ei vector to 1                   
        ei[j] = 1             
        #assign the partial derivative of equations to jth column using the formula                         
        J[:, j] = (f(x + epsilon*ei) - f(x)) / epsilon 
    end
    return J
end
# initial guess 
x0 = [0.1, 0.1, 0.1, 0.1]
# set the value for epsilon
epsilon = 0.000001
# call the function
JacobianMatrix = find_jac(f, x0, epsilon)
function find_root(f, x0, epsilontol, Max_Iter)
     #store initial guess of x0 to x
    x = copy(x0)           
         # set the iteration count to 0 for loop     
    iter = 0      
     #check wether the error between the x and its prevoius value is in the desired tolerance and stop iterations             
    while norm(f(x))> epsilontol && iter < Max_Iter  
         # call function for jacobian matrix 
        J = find_jac(f, x, epsilon)  
        # calculation of step size for jacobian to add to x
        y = -J \ f(x)
        # update the x value by adding the delta to it  
        x += y  
         # add 1 to iter so we can control maximum iterations  
        iter += 1   
    end
    return [x,f(x),norm(f(x))]
end
Final=find_root(f,x0,epsilon,100) # call newton method function
