from pylab import *
from numpy import *
import pymc as pm
from pymc.gp import GPEvaluationGibbs
import model
import imp
imp.reload(model)
import sys
import pickle
from images_plot_dhps_pf import *
from validation_dhps_pf import *
from utils import *


# ================================================================================
#  MCMC run
# ================================================================================
def mcmc(codon):
  # ================================================================================
  #  Load the data and convert from record array to dictionary
  # ================================================================================
  data_loc = '/Users/jflegg/code/dhps_maps_November2020/' ## change as required!
  if codon == 540:
    name = 'dhps540E_for_code_November2020_mod'  ## swap out data as required!
  elif codon == 437:
    name = 'dhps437G_for_code_November2020_mod'  ## swap out data as required!
  elif codon == 581:
    name = 'dhps581G_for_code_November2020_mod'  ## swap out data as required!
  
  data = csv2rec(data_loc+name+'.csv') 
  if data.dtype.names[0]=='\ufeffid':
    data.dtype.names  = tuple(hstack(['id', data.dtype.names[1:8]]))
  data = dict([(k,data[k]) for k in data.dtype.names])


  # ================================================================================
  #   Transform from degrees to radians
  # ================================================================================
  data['lat'], data['lon'] = convert_coords(data['lat'],data['lon'], 'degrees', 'radians')


  # ================================================================================
  #  Get the covariates needed: 'prev'
  # ================================================================================
  # stack the values of lon and lat and year together
  data_mesh = np.vstack((data['lon'],data['lat'], data['year'])).T
  data['pf'] = getCovariatesForLocations(data_mesh)


  # ================================================================================
  # Figure out what the database file is supposed to be
  # ================================================================================
  hf_name = 'res_db.hdf5'
  hf_path,hf_basename = os.path.split(hf_name)
  prev_db = None
  if hf_path=='':
     hf_path='./'
  if hf_basename in os.listdir(hf_path):
     rm_q = input('\nDatabase file %s already exists in path %s. Do you want to continue sampling? [yes or no] '%(hf_basename, hf_path))
     if rm_q.strip() in ['y','YES','Yes','yes','Y']:
         prev_db = pm.database.hdf5.load(os.path.join(hf_path,hf_basename))
     elif rm_q.strip() in ['n','NO','No','no','N']:
         rm_q = input('\nDo you want me to remove the file and start fresh? [yes or no] ')
         if rm_q.strip() in ['y','YES','Yes','yes','Y']:
             print('\nExcellent.')
             os.remove(hf_name)
         elif rm_q.strip() in ['n','NO','No','no','N']:
             raise OSError("\nOK, I'm leaving it to you to sort yourself out.")
         else:
             raise OSError("\nI don't know what you are trying to say.Move, rename or delete the database to continue.")


  # ================================================================================
  #  Create model     
  # ================================================================================
  if prev_db is not None:
     ResistanceSampler = pm.MCMC(model.make_model(**data),db=prev_db,dbmode='a', dbcomplevel=1,dbcomplib='zlib',dbname=hf_name)
  else:
     ResistanceSampler = pm.MCMC(model.make_model(**data),db='hdf5',dbmode='w', dbcomplevel=1,dbcomplib='zlib',dbname=hf_name)


  # ================================================================================
  #   Do the inference 
  # ================================================================================
  N1 = 500000
  N2 = 300000
  N3 = 100
  thin = 20
  first_year = 1990
  

  # set the step methods
  ResistanceSampler.use_step_method(GPEvaluationGibbs, ResistanceSampler.S, 1/ResistanceSampler.Vinv, ResistanceSampler.field_plus_nugget, ti=ResistanceSampler.ti)
  def isscalar(s):
     return (s.dtype != np.dtype('object')) and (np.alen(s.value)==1) and (s not in ResistanceSampler.field_plus_nugget)
  scalar_stochastics = filter(isscalar, ResistanceSampler.stochastics)
  ResistanceSampler.use_step_method(pm.gp.GPParentAdaptiveMetropolis, scalar_stochastics, delay=5000, interval=500)
  ResistanceSampler.step_method_dict[ResistanceSampler.log_sigma][0].proposal_sd *= .1

  # do the sampling
  ResistanceSampler.isample(N1, N2, N3, verbose = 0)  # put verbose = 1 to get all the output.... 


  # ================================================================================
  # Save necessary info to reproduce plotting results and figures to a pickle file
  # ================================================================================
  joint_437_540_581 = 0
  with_pf = 1
  Replot_dict = {'data':data, 'N1':N1, 'N2':N2, 'N3':N3, 'thin':thin, 'first_year':first_year, 'name': name, 'joint_437_540_581':joint_437_540_581,'with_pf':with_pf}
  file = open("Replot_dict.pck", "wb")
  pickle.dump(Replot_dict, file)
  file.close()


  # ================================================================================
  #   Do the plotting of slices, traces, autocorrlations etc 
  # ================================================================================
  plot_slices(data, name,first_year, thin, joint_437_540_581, with_pf)


  # ================================================================================
  #   Do the validataions
  # ================================================================================
  first_year = 2020
  Validate = validate(data,name, N1, N2, N3, thin, first_year, joint_437_540_581, with_pf)
    

# ================================================================================
#  RePlotCommands
# ================================================================================
def RePlotCommands():
  file = open("Replot_dict.pck", "rb") # read mode
  Replot_dict = pickle.load(file)
  name = Replot_dict['name']
  data = Replot_dict['data']
  first_year = Replot_dict['first_year']
  thin = Replot_dict['thin']
  joint_437_540_581 = Replot_dict['joint_437_540_581']
  with_pf = Replot_dict['with_pf']
  
  # ================================================================================
  #   Do plotting of slices, traces, autocorrlations etc, from the save databases 
  # ================================================================================
  plot_slices(data, name,first_year, thin, joint_437_540_581, with_pf)


# ================================================================================
#  RePlotCommands_from_raster
# ================================================================================
def RePlotCommands_from_raster():
  # ================================================================================
  #  Load the data from the pickle file
  # ================================================================================
  file = open("Replot_dict.pck", "rb") # read mode
  Replot_dict = pickle.load(file)
  name = Replot_dict['name']
  data = Replot_dict['data']  
  first_year = Replot_dict['first_year']
  thin = Replot_dict['thin']
  joint_437_540_581 = Replot_dict['joint_437_540_581']
  with_pf = Replot_dict['with_pf']

  plot_from_raster(name, first_year, data,joint_437_540_581, with_pf)


# ================================================================================
#  Do only the mcmc for the validations
# ================================================================================
def mcmc_validations_only():
  #  Load the data from the pickle file
  file = open("Replot_dict.pck", "rb") # read mode
  Replot_dict = pickle.load(file)
  name = Replot_dict['name']
  data = Replot_dict['data']
  first_year = Replot_dict['first_year']
  thin = Replot_dict['thin']
  N1 = Replot_dict['N1']
  N2 = Replot_dict['N2']
  N3 = Replot_dict['N3']
  joint_437_540_581 = Replot_dict['joint_437_540_581']
  with_pf = Replot_dict['with_pf']
  first_year = 2020

  Validate = validate(data,name, N1, N2, N3, thin, first_year, joint_437_540_581, with_pf)


# ================================================================================
#  Redo the validations (bringing them together) from saved MCMC runs
# ================================================================================
def mcmc_validations_bring_together():
  #  Load the data from the pickle file
  file = open("validations_dict.pck", "rb") # read mode
  validations_dict = pickle.load(file)
  name = validations_dict['name']
  data = validations_dict['data']
  thin = validations_dict['thin']
  choice1 = validations_dict['choice1']
  joint_437_540_581 = validations_dict['joint_437_540_581']
  with_pf = validations_dict['with_pf']
  first_year = 2020

  redo_from_saved_validate(data, name, choice1, thin, first_year, joint_437_540_581, with_pf, continent = "Africa")


# ================================================================================
#  Do validation MCMCs one at a time
# ================================================================================
def mcmc_validations_only_one_at_a_time(i):
  #  Load the data from the pickle file
  file = open("validations_dict.pck", "rb") # read mode
  validations_dict = pickle.load(file)
  name = validations_dict['name']
  data = validations_dict['data']
  first_year = validations_dict['first_year']
  thin = validations_dict['thin']
  N1 = validations_dict['N1']
  N2 = validations_dict['N2']
  N3 = validations_dict['N3']
  choice1  = validations_dict['choice1']
  joint_437_540_581 = validations_dict['joint_437_540_581']
  with_pf = validations_dict['with_pf']
  first_year = 2020

  Validate = validate_single_group(data, name, N1, N2, N3, thin, first_year, choice1, i, joint_437_540_581,  with_pf, continent = "Africa")



