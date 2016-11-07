# Cocaine Dependence

Code and data for reproducing key results in the paper "[Utility of Machine-Learning Approaches to Identify Behavioral Markers for Substance Use Disorders: Impulsivity Dimensions as Predictors of Current Cocaine Dependence](http://journal.frontiersin.org/article/10.3389/fpsyt.2016.00034/full)".

Installation
------------

Run the following in a shell:

```shell
git clone https://github.com/CCS-Lab/cocaine-dependence.git
cd cocaine-dependence/R
```

Then run the following in an R terminal:

```r
if (packageVersion("devtools") < 1.6) {
  install.packages("devtools")
}
devtools::install_github("CCS-Lab/easyml", subdir = "R")
```

Getting started
---------------

Run the following in a shell:

```shell
r analysis.R
```
