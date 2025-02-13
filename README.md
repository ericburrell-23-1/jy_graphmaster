# JY_GraphMaster

## Overview

This is an area for Julian to work on GraphMaster. It is a separate repo so that it is clean and only has what is needed.

We will not add extra "fluff" code here. It is barebones by design. No pre-processing. No print statements. Just code that works, and whatever Julian wants to experiment with. He can add that stuff when he feels it is appropriate.

## File Descriptions and Architecture

Here I will give a description of everything that is initially in this directory. Many functions are left blank (`pass`) to be filled in by the team. I will go folder-by-folder.

### algorithm

This contains most of the top-level code associated with solving the general GraphMaster problem. It has several modules.

#### solver.py

This is the entry point to the solver. It is created by each `optimization_problem` instance and its `solve` method is called directly. This is where the top-level algorithm is written. It makes calls to the other modules, such as `restricted_master_problem`, `pricing_problem`, and `StateUpdateFunction`.

#### restricted_master_problem.py

This builds and solves the GM RMP. It returns the dual vector to be used by pricing. This file is mostly empty right now.

#### pricing_problem.py

This solves the RCSPP to find a new column. It does NOT generate new states. It returns the nodes, actions, and total cost of the new path. I filled this in with a VERY BAD version of the pricing. It will need to be replaced with a better pricing function.

#### update_states

This folder is where different *state_update_functions* will be written for different types of problems. It currently contains `state_update_function.py` which is an abstract class, and `standard_CVRP.py` which defines how the *state_update_function* for CVRP will look. That module is not yet complete, but has some code in order to be instructive on how it should look.

### common

This directory contains data structures. Some are very simple, others (such as `multi_state_graph.py`) encapsulate much more logic. That particular module will require a lot of code to be filled in.

### problems

This is where problem-specific modules (our "puzzle pieces") are defined. We have one for CVRP problems defined already. I will make one for LoadAI. We also have `optimization_problem.py`, which is the abstract class defining the structure of each problem.

### test

This is where test files will be written. I have written one to test CVRP. I will also add some test files in an `assests` directory here.



