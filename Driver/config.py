CFG = {
    # simulation parameters
    'task' : 'driver', # the experiment type. Available are driver, driverextended and fetch
    'sigma_values': [.1, .3],
    'sigma_values': [.001],
    'alpha_values': [.5, 1.0, .75, .25],
    'alpha_values': [.5],
    'slider_step_size' : [.1, 1.0, 2.0],
    'slider_step_size' : [2.0],
    'acquisitions' : ['random','regret','information'],
    'acquisitions' : ['random'],

    # plotting parameters
    'path' : 'simulation_data',
    'sigma_plot': 0.001
}
