import numpy as np
from scipy.optimize import minimize


def optimize_wrapper(
        inputs,
        targets,
        init_sigma_mean=10.0,
        init_sigma_variance=0.5,
        num_loops=10000,
):
    num_features, num_stimuli = inputs.shape

    target_mean = np.mean(targets)
    target_std = np.std(targets)

    # Randomize from a normal distribution about given mean.
    init_sigmas = np.abs(
        init_sigma_mean + init_sigma_variance * np.random.randn(num_features, 1)
    )

    gradients = np.zeros_like(init_sigmas)

    optimization_options = {'disp': None,
                            'maxls': 50,
                            'iprint': -1,
                            'gtol': 1e-36,
                            'eps': 1e-8,
                            'maxiter': num_loops,
                            'ftol': 1e-36}

    # Set bounds to 1, 1e15 for all model features.
    optimization_bounds = [
        (1.0 * i, 1e15 * i) for i in np.ones((num_features,))
    ]

    def correlation(sigmas):
        kernel = np.zeros(num_stimuli)

        for i in range(num_stimuli):
            kernel[i] = -np.sum(
                np.power(
                    np.divide(inputs[:, i], sigmas + np.finfo(float).eps),
                    2
                )
            )

        kernel_mean = np.mean(kernel)
        kernel_std = np.std(kernel)

        a = np.sum(
            np.multiply(kernel - kernel_mean, targets - target_mean)
        )

        b = num_stimuli * target_std * kernel_std

        return a / b

    def gradient(sigmas):
        kernel = np.zeros(num_stimuli)
        d_kernel = np.zeros([num_stimuli, len(inputs)])

        for i in range(num_stimuli):
            kernel[i] = -np.sum(
                np.power(
                    np.divide(inputs[:, i], sigmas + np.finfo(float).eps),
                    2
                )
            )
            d_kernel[i, :] = 2 * np.divide(
                np.power(inputs[:, i], 2),
                (np.power(sigmas, 3) + np.finfo(float).eps)
            )

        kernel_mean = np.mean(kernel)
        kernel_std = np.std(kernel)

        Jn = np.sum(
            np.multiply(
                kernel - kernel_mean, targets - target_mean
            )
        )

        Jd = num_stimuli * target_std * kernel_std

        for k in range(num_features):
            tmp = d_kernel[:, k]
            dJn = np.sum(tmp * (targets - target_mean))
            dJd = np.divide(
                target_std * np.sum(tmp * (kernel - kernel_mean)),
                kernel_std + np.finfo(float).eps
            )
            gradients[k] = np.divide(
                Jd * dJn - Jn * dJd,
                np.power(Jd, 2) + np.finfo(float).eps
            )
        return gradients

    def call_back(sigmas):
        # Ideally would save here.
        print(sigmas)

    _ = minimize(correlation, init_sigmas, args=(), method='L-BFGS-B',
                 jac=gradient, callback=call_back,
                 options=optimization_options, bounds=optimization_bounds)


if __name__ == '__main__':
    test_inputs = np.random.rand(10000, 500)
    test_targets = np.random.rand(500)
    optimize_wrapper(test_inputs, test_targets)
