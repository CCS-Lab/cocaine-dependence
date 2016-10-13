sample_equalProp <- function(x, trainProp = 0.667) {
  # trainProp --> proportion of data in training set (e.g., 0.667 = 66.7%)
  # randomly sample individuals for train and test sets, in proportion to group1/group2 ratio
  # e.g., All participants: 30 HCs and 20 users
  # --> 33 subjects in the training set --> 33*30/50 => 20 HCs and 12 users
  #     17 subjects in the test set     --> 33*20/50 => 13 HCs and 8 users
  # x: matrix, which must have "group" column. 0=group1, 1=group2
  # trainProp: proportion of train set (e.g., 66.7%)
  # e.g., 
  # x = data.frame ( addictVar = c( rep(0,20), rep(1,30)), values = 1:50 ); x = as.matrix(x)
  # trainProp = 0.667
  # output = sample_equalProp(x)
  # ouptut[[1]] --> train set (66.7%) 
  # output[[2]] --> test set (33.3%)
  # Currently, group variable should be 'addictVar'
  # Programmed by Woo-Young Ahn
  
  testProp = 1 - trainProp # proportion of data in test set 
  numSubjs = dim(x)[1]
  
  group1_index = which(x[, "addictVar"]==0)  # row number
  group2_index = which(x[, "addictVar"]==1)  # row number

  group1_train_n = round(length(group1_index)*trainProp)  # number of group1 subjects in the train set
  group1_test_n = length(group1_index) - group1_train_n   # number of group1 subjects in the test set
  group2_train_n = round(length(group2_index)*trainProp)  # number of group1 subjects in the train set
  group2_test_n = length(group2_index) - group2_train_n   # number of group1 subjects in the test set
  
  group1_train_index = sample(group1_index, group1_train_n, replace = F)  # train <-- sample from group1, N= ...
  group1_test_index =  group1_index[!group1_index %in% group1_train_index] # all group subjects except group1_train..
  
  group2_train_index = sample(group2_index, group2_train_n, replace = F)  # train <-- sample from group1, N= ...
  group2_test_index =  group2_index[!group2_index %in% group2_train_index] # all group subjects except group1_train..

  all_train_index = sort( c(group1_train_index, group2_train_index) )
  all_test_index = sort( c(group1_test_index, group2_test_index) )
  
  all_train = x[all_train_index, ]
  all_test = x[all_test_index, ]
  
  output = list(all_train, all_test) # output[[1]] --> train set, output[[2]] --> test set
  
  return(output)
}
