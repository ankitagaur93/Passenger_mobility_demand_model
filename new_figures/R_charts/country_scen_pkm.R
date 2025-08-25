library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork) 
library(grid)
library(cowplot)
library(ggpubr)
library(stringr) 
library(glue)


scen <- c("Base", "NP", "BP", "TOD", "TECH")
path <- "C:/Users/agaur/Passenger_mobility_demand_model/"

all_data <- list()

for (i in scen) 
  {
  filename <- paste0(path, "total_pdt_", i, "_1_1.csv")
  dat <- read.csv(filename)
  dat <- dat %>% filter(y %in% c(2011, 2050))
  dat$scen <- i
  all_data[[length(all_data) + 1]] <- dat
}

new_dat <- bind_rows(all_data)

# Remove 2011 rows for certain scenarios
new_dat <- new_dat %>%
  filter(!(y == 2011 & scen %in% c("BP", "NP", "TOD", "TECH")))

# Group by and sum values
new_dat <- new_dat %>%
  group_by(n, y, area_type, scen) %>%
  summarise(value = sum(value, na.rm = TRUE), .groups = "drop")

new_dat$value <- new_dat$value/10^3

change<- new_dat %>%
  group_by(n, area_type) %>% 
  mutate(
    base2011 = value[y == 2011 & scen == "Base"],  # take 2011 Base as reference
    change = ifelse(y == 2050, (value / base2011 - 1), NA)
  ) %>%
  ungroup()

dat_alt <-new_dat %>% filter(y==2050)
btw_scn<-dat_alt%>%
  group_by(n, area_type) %>% 
  mutate(
    base = value[scen == "Base"],  # take 2011 Base as reference
    btw_scn = ifelse(scen != "Base", (value / base - 1), NA)
  )

country_list<- unique(new_dat$n)


combined_plots_list<-list()
for (i in country_list)
  {

  df<-filter(new_dat, n==i)
  df$area_type<-factor(df$area_type, levels=c("large_city", "city", "town", "rural"))
  
  df_2011 <- df %>%
    filter(y == 2011)
  
  df_2050 <- df %>%
    filter(y == 2050)
  # Combine them back
  
  # Sum values per scenario for 2050
  max_height_2050 <- df_2050 %>%
    group_by(scen) %>%
    summarise(total_value = sum(value, na.rm = TRUE)) %>%
    summarise(max_total = max(total_value)) %>%
    pull(max_total)
  
  # Set y-limits from 0 to max total stacked height in 2050
  y_limits <- c(0, max_height_2050)
  
  
  plot_2011 <- ggplot(df_2011, aes(x = scen, y = value, fill = area_type)) +
    geom_bar(stat = "identity", position = "stack", width=0.4) +
    scale_fill_manual(values=c("city"= "#14213d",
                               "large_city"= "#fca311",
                               "town"= "#fb6f92",
                               "rural"="#2a9d8f"))+
    labs(, x = NULL, y = "Bpkm", fill = "") +
    coord_cartesian(ylim = y_limits) +
    theme_light() +
    theme(axis.text.x = element_text(color="white"),   # hide x axis text because scen is empty
          #axis.ticks.x = element_blank(),
          legend.position = "none",
          axis.text=element_text(face="bold",size=32),
          axis.title = element_text(face="bold",size=32))
  
  plot_2050 <- ggplot(df_2050, aes(x = scen, y = value, fill = area_type)) +
    geom_bar(stat = "identity", position = "stack", width=0.5) +
    scale_fill_manual(values=c("city"= "#14213d",
                               "large_city"= "#fca311",
                               "town"= "#fb6f92",
                               "rural"= "#2a9d8f"))+
    labs( x = "", y = NULL, fill = "") +
    theme_light()+theme(axis.text.y = element_blank(),
                        axis.ticks.y = element_blank(),
                        axis.title.y = element_blank(),
                        plot.margin = margin(t = 1, r = 1, b = 1, l = 1),
                    
                        legend.position = "none",
                        axis.text=element_text(face="bold",size=32))
                        
  
   # for str_wrap
  
  # Your label text, wrapped at ~40 characters (adjust as needed)
  label_text <- str_wrap("2050", width = 30)
  
  box_grob_2050 <- ggdraw() +
    draw_plot(plot_2050, x = 0, y = 0.06, width = 1, height = 0.93) +
    draw_grob(
      grobTree(
        rectGrob(
          x = 0.5, y = 0.015, width = unit(0.95, "npc"), height = unit(0.055, "npc"),
          gp = gpar(fill = "white", col = NA)
        )
        
      )
    )
  
  
  box_grob_2011 <- ggdraw() +
    draw_plot(plot_2011, x = 0, y = 0.08, width = 1, height = 0.918) +
    draw_grob(
      grobTree(
        rectGrob(
          x = 0.6, y = 0.015, width = unit(0.7, "npc"), height = unit(0.07, "npc"),
          gp = gpar(fill = "white", col = NA)
        )
        )
      )
    
  box_grob_2011
  
  combined <- ggarrange(
    box_grob_2011, box_grob_2050,
    ncol         = 2,
    widths       = c(1, 3))
  # Add a figure-level title above the combined plot
  title_text <- paste0(i)
  title_grob <- ggdraw() +
    draw_label(title_text, fontface = "bold", size = 36,
               x = 0.2, hjust = 0.5, vjust = 0.1)
  
  combined <- plot_grid(
    title_grob, combined,
    ncol = 1,
    rel_heights = c(0.08, 0.5)  # adjust the top space as needed
  )

  
  combined_plots_list[[i]] <- combined
  
  
  
}


# all_combined <- ggarrange(
#   combined_plots_list[["Afghanistan" ]],
#   combined_plots_list[["Bangladesh" ]],
#   combined_plots_list[["Bhutan" ]],
#   combined_plots_list[["India" ]],
#   combined_plots_list[["Maldives" ]],
#   combined_plots_list[["Nepal" ]],
#   combined_plots_list[["Pakistan" ]],
#   combined_plots_list[["Sri Lanka"]],
#   ncol         = 2,
#   nrow=8,
#   widths       = c(1, 1))
# 
# ggsave( filename = "SouthAsia_all_combined.pdf", plot = all_combined, 
#         device = "pdf", width = 14, height = 20, units = "in" )

