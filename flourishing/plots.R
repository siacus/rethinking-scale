# Plots of the results of the analysis
# (S.M.Iacus 2024)

library(data.table)
library(ggplot2)
library(ggrepel)

library(patchwork)

x <- read.csv("summaryOld.csv")
x$name <- sprintf("%s-%sB%s",x$model,x$bpar,ifelse(x$ft=="YES", "-FT",""))
x$mtype <- sprintf("%s%s",x$model,ifelse(x$ft=="YES", "-FT",""))

y <- read.csv("verifyOld.csv")
y$name <- sprintf("%s-%sB%s",y$model,y$bpar,ifelse(y$ft=="YES", "-FT",""))
y$mtype <- sprintf("%s%s",y$model,ifelse(y$ft=="YES", "-FT",""))


pAccD <- ggplot(x, aes(x=bpar, y=accD,colour = mtype)) + 
  geom_point(size=2) +
  geom_line() +
  geom_text_repel(label=x$name, segment.color = 'black',nudge_x = 0.3, max.overlaps = 30)+
  xlab("log(Billion parameters)") + 
  ylab("Accuracy on Dimension") +
  scale_x_continuous(trans='log10')+
  guides(colour=guide_legend(title="Model"))+
  scale_y_continuous(labels = scales::percent, limits = c(0.2,1))+
  theme(legend.position = "none") + 
  ggtitle("Training set")


pAccDT <- ggplot(y, aes(x=bpar, y=accD,colour = mtype)) + 
  geom_point(size=2) +
  geom_line() +
  geom_text_repel(label=y$name, segment.color = 'black',nudge_x = 0.3, max.overlaps = 30)+
  xlab("log(Billion parameters)") + 
  ylab("Accuracy on Dimension") +
  scale_x_continuous(trans='log10')+
  guides(colour=guide_legend(title="Model"))+
  scale_y_continuous(labels = scales::percent, limits = c(0.2,1))+ #c(0.0,0.3))+
  theme(legend.position = "none") + 
  ggtitle("Test set")

pAccD+pAccDT+plot_layout(guides = "collect")


pAccI <- ggplot(x, aes(x=bpar, y=accI,colour = mtype)) + 
  geom_point(size=2) +
  geom_line() +
  geom_text_repel(label=x$name, segment.color = 'black',nudge_x = 0.3)+
  xlab("log(Billion parameters)") + 
  ylab("Accuracy on Intensity | Dimension") + 
  scale_x_continuous(trans='log10')+
  guides(colour=guide_legend(title="Model"))+
  scale_y_continuous(labels = scales::percent, limits = c(0.25,1))+
  theme(legend.position = "none") + 
  ggtitle("Training set")


pAccIT <- ggplot(y, aes(x=bpar, y=accI,colour = mtype)) + 
  geom_point(size=2) +
  geom_line() +
  geom_text_repel(label=y$name, segment.color = 'black',nudge_x = 0.3)+
  xlab("log(Billion parameters)") + 
  ylab("Accuracy on Intensity | Dimension") + 
  scale_x_continuous(trans='log10')+
  guides(colour=guide_legend(title="Model"))+
  scale_y_continuous(labels = scales::percent, limits = c(0.25,1))+
  theme(legend.position = "none") + 
  ggtitle("Test set")


pAccI+pAccIT+plot_layout(guides = "collect")


pJac <- ggplot(x, aes(x=bpar, y=jac2,colour = mtype)) + 
  geom_point(size=2) +
  geom_line() +
  geom_text_repel(label=x$name, segment.color = 'black',nudge_x = 0.3)+
  xlab("log(Billion parameters)") + 
  ylab("Jaccard Index (the higher, the better)") +
  scale_x_continuous(trans='log10')+
  guides(colour=guide_legend(title="Model"))+
  scale_y_continuous(labels = scales::percent, limits  = c(0.25,1))+ #rrange(x$jac2))+ #c(0.0,0.3))+
  theme(legend.position = "none") + 
  ggtitle("Training set")


pJacT <- ggplot(y, aes(x=bpar, y=jac2,colour = mtype)) + 
  geom_point(size=2) +
  geom_line() +
  geom_text_repel(label=y$name, segment.color = 'black',nudge_x = 0.3)+
  xlab("log(Billion parameters)") + 
  ylab("Jaccard Index (the higher, the better)") +
  # ylim(0.15,0.7)+
  scale_x_continuous(trans='log10')+
  guides(colour=guide_legend(title="Model"))+
  scale_y_continuous(labels = scales::percent, limits = c(0.25,0.9))+ #rlimits = range(y$jac2))+ #c(0.0,0.3))+
  theme(legend.position = "none") + 
  ggtitle("Test set")

pJac+pJacT+plot_layout(guides = "collect")



pHam <- ggplot(x, aes(x=bpar, y=ham,colour = mtype)) + 
  geom_point(size=2) +
  geom_line() +
  geom_text_repel(label=x$name, segment.color = 'black',nudge_x = 0.3)+
  xlab("log(Billion parameters)") + 
  ylab("Hamming loss (the lower, the better)") + 
  scale_x_continuous(trans='log10')+
  guides(colour=guide_legend(title="Model"))+
  scale_y_continuous(labels = scales::number, limits = c(0,0.8))+
  theme(legend.position = "none") + 
  ggtitle("Training set")


pHamT <- ggplot(y, aes(x=bpar, y=ham,colour = mtype)) + 
  geom_point(size=2) +
  geom_line() +
  geom_text_repel(label=y$name, segment.color = 'black',nudge_x = 0.3)+
  xlab("log(Billion parameters)") + 
  ylab("Hamming loss (the lower, the better)") + 
  #  ylim(0.15,0.7)+
  scale_x_continuous(trans='log10')+
  guides(colour=guide_legend(title="Model"))+
  scale_y_continuous(labels = scales::number, limits = c(0,0.8))+
  theme(legend.position = "none") + 
  ggtitle("Test set")

pHam+pHamT+plot_layout(guides = "collect")


