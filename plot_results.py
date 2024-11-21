import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    pass  # pass is just a placeholder if there is no other code

    def main():
        # Load the CSV file
        file_path = "mia-results/results.csv"  # Adjust the path if needed
        data = pd.read_csv(file_path)
        
        # Ensure metrics are recognized dynamically
        metrics = [col for col in data.columns if col not in ['SUBJECT', 'LABEL']]
        
        # Generate plots for each metric
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=data, x="LABEL", y=metric, palette="Set3")
            plt.title(f'{metric} Distribution per Label')
            plt.xticks(rotation=45)
            plt.ylabel(metric)
            plt.xlabel("Label")
            plt.tight_layout()
            plt.show()
            # Optionally save the plot:
            plt.savefig(f"{metric}_boxplot.png")


if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
