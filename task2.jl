using LinearAlgebra

#make the initial function
function f(v)
    # Calculate the length of the input vector
    n = length(v)

    # Calculate the step size
    h = 1 / (n + 1)

    # Initialize the residual vector with zeros
    f = zeros(n)

    # Compute the first and last elements of the residual vector
    f[1] = -2 * v[1] + v[2] + (1 / 4) * exp(v[1])
    f[n] = -2 * v[n] + v[n-1] + (h^2) * exp(v[n])

    # Loop through the elements of v except for the first and last
    for i in 2:n-1
        # Compute the i-th element of the residual vector
        f[i] = ((v[i+1] - 2 * v[i] + v[i-1]) / h^2) + exp(v[i])

        # Multiply the i-th element of the residual vector by h^2
        f[i] *= h^2
    end

    # Return the residual vector
    return f
end


epsilon=0.000001

function exact_jacobian(v)
    # Calculate the length of the input vector
    n = length(v)

    # Calculate the step size
    h = 1 / (n + 1)

    # Initialize the Jacobian matrix with zeros
    J = zeros(n, n)

    # Compute the first and last rows of the Jacobian matrix
    J[1, 1] = -2 + (1/4) * exp(v[1])
    J[1, 2] = 1
    J[n, n-1] = 1
    J[n, n] = -2 + h^2 * exp(v[n])

    # Loop through the elements of v except for the first and last
    for i in 2:n-1
        # Calculate the step size for the loop
        h_i = 1 / (i+1)

        # Set the values for the i-th row of the Jacobian matrix
        J[i, i-1] = 1
        J[i, i] = -2 + (h_i^2) * exp(v[i])
        J[i, i+1] = 1
    end

    # Return the Jacobian matrix
    return J
end
function find_jacobian_matrix(f, x, epsilon)
    # Get the length of the input vector
    n = length(x)

    # Initialize the Jacobian matrix with zeros
    J = zeros(n, n)

    # Loop through the columns of the Jacobian matrix
    for j in 1:n
        # Create an elementary vector with a 1 in the jth position and 0s elsewhere
        ei = zeros(n)
        ei[j] = 1

        # Calculate the jth column of the Jacobian matrix by taking the finite difference approximation
        # using the elementary vector and the input vector
        J[:, j] = (f(x + epsilon*ei) - f(x)) / epsilon
    end

    # Return the Jacobian matrix
    return J
end
function newton_method(f, x0, epsilon, max_iter, method::Symbol=:exact)
    # Make a copy of the initial guess to avoid modifying it
    x = copy(x0)

    # Initialize the iteration counter
    iter = 0

    # Keep iterating until the norm of the function values is less than epsilon or the maximum number of iterations is reached
    while norm(f(x)) > epsilon && iter < max_iter
        # Compute the Jacobian matrix using the specified method
        if method == :exact
            J = exact_jacobian(x)
        elseif method == :approximate
            J = find_jacobian_matrix(f, x, epsilon)
        else
            error("Invalid method. Must be :exact or :approximate.")
        end

        # Compute the Newton update direction by solving the linear system Jy = -f(x)
        y = -J \ f(x)

        # Update the current guess by adding the Newton update
        x += y

        # Increment the iteration counter
        iter += 1
    end

    # Return the final guess
    return x
end


# initial guess 
n=2
v=zeros(n)

# call the function
exact_root = newton_method(f,v,epsilon,50,:exact)
approx_root = newton_method(f, v, epsilon, 50, :approximate)


jac=find_jacobian_matrix(f,v,epsilon)
jac2=exact_jacobian(v)
display(jac)
display(jac2)

# display the results
println("Exact Solution: ", exact_root)
println("Approximate Solution: ", approx_root)