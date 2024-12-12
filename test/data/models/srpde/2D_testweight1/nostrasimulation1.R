# SIMULATION 1 DA CAVAZZUTI TRADOTTA NELLA NOSTRA LIBRERIA 
# libreria nuova con inferenza che abbiamo implementato noi 
library(MASS)
library(plyr)
library(Matrix)
library(ggplot2)
library(RandomFieldsUtils)
library(RandomFields)
library(reshape2)
library(viridis)

library(fdaPDE2)

unit_square <- MeshUnitSquare(n=10)
Vh <- FunctionalSpace(unit_square, type = "fe", order = 1)
f <- Function(Vh)

# per plottare le mesh potrebbe servire questo 
plot(unit_square$nodes, main = "Mesh originale", asp = 1, pch = 16, col = "blue")
for (i in 1:nrow(unit_square$elements)) {
  polygon(unit_square$nodes[unit_square$elements[i, ], ], border = "black")
}


# simulate data by adding normal error
# Locations generation for beta
n_loc <- 225 # original # 225
locations <- matrix(data=NA,nrow=n_loc,ncol=2)
set.seed(543678)
locations[,1]<-runif(n_loc,0,1)
locations[,2]<-runif(n_loc,0,1)

epsilon=n_loc*10^-4


# Plot locations
# per plottare le mesh potrebbe servire questo 
plot(unit_square$nodes, main = "Mesh originale", asp = 1, pch = 16, col = "blue")
for (i in 1:nrow(unit_square$elements)) {
  polygon(unit_square$nodes[unit_square$elements[i, ], ], border = "black")
}
points(locations, col='red', pch=16)


# Function generation:
# Function 2 in gamSim form mgcv package [2017]
#library(mgcv)
gamSim_2 <- function(x,y) {
  (0.4*pi^0.3)*(1.2*exp( - ((x-0.2)^2)/(0.3^2) - ((y-0.3)^2)/(0.4^2)) + 0.8*exp( - ((x-0.7)^2)/(0.3^2) - ((y-0.8)^2)/(0.4^2)))
}

exact_data <- as.matrix(gamSim_2(locations[, 1], locations[, 2]), ncol = 1)

# Covariates (random fields) definition
# Gaussian random field with mean=0, scale=0.05
set.seed(1)
model <- RMgauss(scale = 0.05)
cov_1 <- RandomFields::RFsimulate(model, x = SpatialPoints(locations), n = 1)@data
cov_1 <- unlist(cov_1)
plot(cov_1)
# image(FEM(coeff = unlist(RandomFields::RFsimulate(model, x = SpatialPoints(mesh$nodes), n = 1)@data), FEMbasis = FEMbasis))
# rgl.snapshot(filename = "square_Covariate_1.png")

# Matérn random field with nu=1, sigma=2 and scale=0.1
set.seed(2)
model <- RMmatern(nu = 1, var = 2, scale = 0.1)
cov_2 <- RandomFields::RFsimulate(model, x = SpatialPoints(locations), n = 1)@data
cov_2 <- unlist(cov_2)
plot(cov_2)
# image(FEM(coeff = unlist(RandomFields::RFsimulate(model, x = SpatialPoints(mesh$nodes), n = 1)@data), FEMbasis = FEMbasis))
# rgl.snapshot(filename = "square_Covariate_2.png")

# Deterministic field + Gaussian random field with mean=0, scale=0.05
set.seed(3)
model <- RMgauss(scale = 0.05)
fun_3 <- function(x,y){
  cos(5*(x+y))-(2*x-x*y^2)^2
}
add_3 <- RandomFields::RFsimulate(model, x = SpatialPoints(locations), n = 1)@data
add_3 <- unlist(add_3)
cov_3 <- fun_3(locations[,1],locations[,2]) + add_3
plot(cov_3)
# image(FEM(coeff = fun_3(mesh$nodes[,1], mesh$nodes[,2])+unlist(RandomFields::RFsimulate(model, x = SpatialPoints(mesh$nodes), n = 1)@data), FEMbasis = FEMbasis))
# rgl.snapshot(filename = "square_Covariate_3.png")

# Deterministic field + Matérn random field with nu=1, sigma=2 and scale=0.1
set.seed(4)
model <- RMmatern(nu = 1, var = 2, scale = 0.1)
fun_4 <- function(x,y){
  cos(5*(x+y))-(2*x-x*y^2)^2
}
add_4 <- RandomFields::RFsimulate(model, x = SpatialPoints(locations), n = 1)@data
add_4 <- unlist(add_4)
cov_4 <- fun_4(locations[,1],locations[,2]) + add_4
plot(cov_4)
# image(FEM(coeff = fun_4(mesh$nodes[,1], mesh$nodes[,2])+unlist(RandomFields::RFsimulate(model, x = SpatialPoints(mesh$nodes), n = 1)@data), FEMbasis = FEMbasis))
# rgl.snapshot(filename = "square_Covariate_4.png")

# Setting Parameters for smoothing.R
lambda = 10^seq(-3,3,by=0.25)

# Set beta_0
beta_0 = 0.0




#### Notice: add time estimated when done : 13 s for 20 simulations
# Number of repetitions for each simulation
rep = 2 # 1000 for paper

# List of hypotheses under which data are generated
beta_H1_list <- beta_0+ seq(from = 0, by = 0.01, length.out = 11)

# Scale f for simulations (Also covariates need to be scaled)
#f <- scale(f)

# Plot f on white backgorund
'''plot.mesh.2D_local<-function(x, ...)
{
  plot(x$nodes, xlab="", ylab="", xaxt="n", yaxt="n", bty="n", ...)
  segments(x$nodes[x$edges[,1],1], x$nodes[x$edges[,1],2],
           x$nodes[x$edges[,2],1], x$nodes[x$edges[,2],2], col='azure2',...)
  segments(x$nodes[x$segments[,1],1], x$nodes[x$segments[,1],2],
           x$nodes[x$segments[,2],1], x$nodes[x$segments[,2],2], col="red", ...)
}

rbPal <- colorRampPalette(c('red','yellow'))
plot.mesh.2D_local(mesh, pch='.',asp=1, main='f field')
points(locations, col=rbPal(100)[as.numeric(cut(f,breaks = 100))], pch=16, cex=1.5, axes=F, xlab='', ylab='', asp=1)

# covariates
rbPal <- colorRampPalette(c('red','yellow'))
plot.mesh.2D_local(mesh, pch='.',asp=1)
points(locations, col=rbPal(100)[as.numeric(cut(cov_4,breaks = 100))], pch=16, cex=1.5, axes=F, xlab='', ylab='', asp=1)
'''

# Define the standard deviation of epsilon_1,...,epsilon_n_loc
sd <- 0.1

# Real simulations: 1
covariates <- cov_1
covariates <- scale(covariates)

res_1_ex <- list()
res_1_non_ex <- list()
rmse_1_fdaPDE <- 0
rep=2

for (i in 1:length(beta_H1_list)) {
  ps_ex = matrix(data = NA, nrow = 3, ncol = rep)
  ps_non_ex = matrix(data = NA, nrow = 3, ncol = rep)
  
  for (k in 1:rep) {
    set.seed(k)
    rand= rnorm(n_loc, 0, sd = sd)
    model <- RMgauss(scale = 0.05)
    covariates <- scale(unlist(RandomFields::RFsimulate(model, x = SpatialPoints(locations), n = 1)@data))
    
    observations <- covariates * beta_H1_list[1] + exact_data + rand # Build observations in H1

    data <- data.frame(
      y = observations,
      cov1 = covariates
    )
    
    model <- SRPDE(y ~ f + cov1, data = data)
    model$fit(
      calibration = 1e-2
    )
    model$f
    # Debug prints
    print("Before inference: wald")
    model$inference("wald", "one_at_the_time", "exact", C = matrix(c(1), nrow = 1, ncol = 1), beta0 = 0)
    ps_ex[1, k] <- model$pvalues
    
    print("Before inference: speckman")
    model$inference("speckman","one_at_the_time", "exact", C = matrix(c(1), nrow = 1, ncol = 1), beta0 = beta_0)
    ps_ex[2, k] <- model$pvalues
    
    print("Before inference: esf")
    model$inference("esf","one_at_the_time", "exact", C = matrix(c(1), nrow = 1, ncol = 1), beta0 = beta_0)
    ps_ex[3, k] <- model$pvalues
    #rmse_1_fdaPDE <- rmse_1_fdaPDE + Local_Solution_ex$solution$rmse[Local_Solution_ex$optimization$lambda_position] + Local_Solution_non_ex$solution$rmse[Local_Solution_non_ex$optimization$lambda_position]
  }
  res_1_ex[[i]] <- ps_ex
  #res_1_non_ex[[i]] <- ps_non_ex
  
  print(i)
 # Sys.sleep(2)
}

#rmse_1_fdaPDE <- rmse_1_fdaPDE/(2*rep*length(beta_H1_list))

# Rimuovi o correggi le opzioni non valide
options(list.len = NULL, deparse.lines = NULL)

# Imposta opzioni valide
options(list.len = 99) # Imposta un valore valido per list.len se necessario
options(deparse.lines = 10) # Imposta un valore valido per deparse.lines se necessario







## test function
test_function <- function(x, y, z = 1) {
  coe <- function(x, y) 1 / 2 * sin(5 * pi * x) * exp(-x^2) + 1
  return(sin(2 * pi * (coe(y, 1) * x * cos(z - 2) - y * sin(z - 2))) *
           cos(2 * pi * (coe(y, 1) * x * cos(z - 2 + pi / 2) + coe(x, 1) * y * sin((z - 2) * pi / 2))))
}
exact_data <- as.matrix(test_function(unit_square$nodes[, 1], unit_square$nodes[, 2]), ncol = 1)


# Function generation:
# Function 2 in gamSim form mgcv package [2017]
#library(mgcv)
gamSim_2 <- function(x,y) {
  (0.4*pi^0.3)*(1.2*exp( - ((x-0.2)^2)/(0.3^2) - ((y-0.3)^2)/(0.4^2)) + 0.8*exp( - ((x-0.7)^2)/(0.3^2) - ((y-0.8)^2)/(0.4^2)))
}

# Check similarity
f <- gamSim_2(locations[,1],locations[,2])

# Covariates (random fields) definition
# Gaussian random field with mean=0, scale=0.05
set.seed(1)
model <- RMgauss(scale = 0.05)
cov_1 <- RandomFields::RFsimulate(model, x = SpatialPoints(locations), n = 1)@data
cov_1 <- unlist(cov_1)


# Setting Parameters for smoothing.R
lambda = 10^seq(-3,3,by=0.25)

# Set beta_0
beta_0 = 0.0

# Set inference global objects: one exact and the other non-exact. We test against the null hypothesis beta = beta_0
#Inference_beta_ex <- inferenceDataObjectBuilder(test=c('oat'), type = c('w', 's', 'esf', 'enh-esf'), exact=T, beta0 = beta_0, dim=2, n_cov = 1)


# Number of repetitions for each simulation
rep = 10 # 1000 for paper

# List of hypotheses under which data are generated
beta_H1_list <- beta_0+ seq(from = 0, by = 0.01, length.out = 11)

# Scale f for simulations (Also covariates need to be scaled)
f <- scale(f)

# Define the standard deviation of epsilon_1,...,epsilon_n_loc
sd <- 0.1

# Real simulations: 1
covariates <- cov_1
covariates <- scale(covariates)

res_1_ex <- list()
#res_1_non_ex <- list()
#rmse_1_fdaPDE <- 0

for (i in 1:length(beta_H1_list)) {
  ps_ex = matrix(data = NA, nrow = 3, ncol = rep)
  #ps_non_ex = matrix(data = NA, nrow = 4, ncol = rep)
  
  for (k in 1:rep) {
    set.seed(k)
    rand= rnorm(n_loc, 0, sd = sd)
    model <- RMgauss(scale = 0.05)
    cov_1 <- RandomFields::RFsimulate(model, x = SpatialPoints(locations), n = 1)@data
    covariates <- unlist(cov_1)    
    f_scale=scale(gamSim_2((locations[,1], locations[,2])))
    observations <- covariates * beta_H1_list[i] + f_scale + rand # Build observations in H1
    
    
    #Local_Solution_ex <- smooth.FEM(locations = locations, observations=observations,
     #                               covariates = covariates,
      #                              FEMbasis=FEMbasis, lambda=lambda,
       #                             lambda.selection.criterion='grid', lambda.selection.lossfunction = "GCV", DOF.evaluation = "exact",
        #                            inference.data.object = Inference_beta_ex)
    model_wald <- SRPDE(y ~ f + cov1, data = exact_data)
    model$fit(
      calibration = 1e-2
    )
    Local_solution_ex_wald<- model_wald$inference("wald", "exact",1, beta0 = 0)
    model_speck <- SRPDE(y ~ f + cov1, data = exact_data)
    model$fit(
      calibration = 1e-2
    )
    Local_solution_ex_speck<- model_speck$inference("Speckman", "exact",1, beta0 = 0)
    model_ESF <- SRPDE(y ~ f + cov1, data = exact_data)
    model$fit(
      calibration = 1e-2
    )
    Local_solution_ex_ESF<- model_ESF$inference("ESF", "exact",1, beta0 = 0)
    
    #Local_Solution_non_ex <- smooth.FEM(locations = locations, observations=observations,
     #                                   covariates = covariates,
      #                                  FEMbasis=FEMbasis, lambda=lambda,
       #                                 lambda.selection.criterion='grid', lambda.selection.lossfunction = "GCV", DOF.evaluation = "exact",
        #                                inference.data.object = Inference_beta_non_ex)
    
  #  rmse_1_fdaPDE <- rmse_1_fdaPDE + Local_Solution_ex$solution$rmse[Local_Solution_ex$optimization$lambda_position] + Local_Solution_non_ex$solution$rmse[Local_Solution_non_ex$optimization$lambda_position]
    
    ps_ex[, k] <- c(model_wald$pvalues, model_speck$pvalues, model_ESF$pvalues)
   # ps_non_ex[, k] <- c(Local_Solution_non_ex$inference$beta$p_values$wald[[1]], Local_Solution_non_ex$inference$beta$p_values$speckman[[1]], Local_Solution_non_ex$inferenc$beta$p_values$eigen_sign_flip[[1]],  Local_Solution_non_ex$inference$beta$p_values$enh_eigen_sign_flip[[1]])
    #rmse_1_fdaPDE <- rmse_1_fdaPDE + Local_Solution_ex$solution$rmse[Local_Solution_ex$optimization$lambda_position] + Local_Solution_non_ex$solution$rmse[Local_Solution_non_ex$optimization$lambda_position]
  }
  res_1_ex[[i]] <- ps_ex
  #res_1_non_ex[[i]] <- ps_non_ex
  
  print(i)
  Sys.sleep(2)
}

#rmse_1_fdaPDE <- rmse_1_fdaPDE/(2*rep*length(beta_H1_list))



# Results
# Change apply's input to obtain the different possible results
power_mat <- matrix(NA, nrow = 11, ncol = 3)
for(i in 1:11) {
  power_mat[i, ] <- apply(res_1_ex[[i]], 1, function(x) mean(x < 0.05))[c(1,2,3,4)]
}
t(power_mat)

#### Plotting ####
# Plotting with ggplot2
# genera n colori
gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

#genero 4 colori perchè le colonne di power_mat sono 4
cols <- gg_color_hue(ncol(power_mat))
cols <- cols[c(1,4,2,3)]

colnames(power_mat) <- c("Wald", "Speck", "ESF", "PESF")
dat <- melt(power_mat) # cambia il formato dell matrice ma è sempre la stessa power_mat
dat$Var1 <- rep(beta_H1_list, length(cols)) # inserisco come prima colonna di dat i valori di beta 1  
colnames(dat) <- c("beta", "Test", "value") # quindi chiama le colonne beta, test e value 

dat$Test <- factor(dat$Test, levels = c("Wald", "Speck", "ESF", "PESF")) # ome seconda colonna inserisco i nomi 
#dat$Test <- factor(dat$Test, levels = c("SESF","ESF","Wald", "Speck"))

# plotta sulle x i beta1  e sulle y la probabilità di aver commesso errore di 1 primo ovvero se il pvalue è <0.05
# 
plot_3<-ggplot(dat, aes(x = beta, y = value, colour = Test, linetype = Test, shape = Test)) +
  geom_abline(intercept = 0.05, slope = 0, linetype = "dashed") +
  geom_line(size = 1.5) + geom_point(size = 4) +
  theme_bw() +
  scale_shape_manual(values = c(1, 3, 15, 16)) +
  scale_linetype_manual(values=c("dotted", "dotdash","dashed","solid")) +
  scale_color_manual(values = cols) +
  ggtitle("(d)")+
  theme(plot.title = element_text(hjust = 0.5))+
  #theme(legend.key.width = unit(2,"cm")) +
  theme(legend.position = c(0.77, 0.36)) + ylim(c(-0.01, 1.01)) +
  #scale_y_continuous(breaks=c(0, 0.05, 0.25, 0.5, 0.75, 1))
  theme(legend.title = element_text(size = 20, face = "bold"), legend.text=element_text(size=20)) +
  theme(plot.title = element_text(size = 20, face = "bold")) + #theme(legend.position = "top") +
  theme(axis.text=element_text(size = 12), axis.title=element_text(size=20)) +
  theme(axis.text.x = element_text(size = 20), axis.text.y = element_text(size = 20)) +
  xlab(expr(beta)) + ylab("Power") #+ annotate("text", x = 0.19, y = 0.03, label = "0.05")

plot_3








set.seed(7893475)
data <- data.frame(
  y = exact_data + rnorm(nrow(exact_data), mean = 0, sd = 0.05 * abs(diff(range(exact_data)))),
  cov1 = cov1,
  cov2 =cov2
)

model <- SRPDE(y ~ f + cov1 + cov2, data = data)
model$fit(
  calibration = 1e-2
)
risultati=model$inference("wald", "exact", matrix(c(1,0,0,1), nrow = 2, ncol = 2), beta0 = c(2,-1))
model$pvalues
model$confidence_interval

