import sys
from dhps_run_functions import * 

# Run 437 model and validations
mcmc(437) 

# Run only validation mcmc, but all 10 of them
mcmc_validations_only() 

# Run only validation mcmc, one of the 10 groups at once
i=0 # i = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
mcmc_validations_only_one_at_a_time(i) 

# RePlotCommands_from_raster
RePlotCommands_from_raster() 

# RePlotCommands (main mcmc figures)
RePlotCommands()

#  Redo the validations (bringing them together) from saved MCMC runs
mcmc_validations_bring_together()