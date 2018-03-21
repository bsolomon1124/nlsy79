scp -rp /Users/brad/Scripts/python/metis/metisgh/projects/kojak/nlsy79/src/ brad@$(echo $(ipaws) | sed -e 's/^"//' -e 's/"$//'):/home/brad/nlsy79/src/
scp -rp /Users/brad/Scripts/python/metis/metisgh/projects/kojak/nlsy79/downloads/ brad@$(echo $(ipaws) | sed -e 's/^"//' -e 's/"$//'):/home/brad/nlsy79/downloads/
scp -rp /Users/brad/Scripts/python/metis/metisgh/projects/kojak/nlsy79/plots/ brad@$(echo $(ipaws) | sed -e 's/^"//' -e 's/"$//'):/home/brad/nlsy79/plots/
scp -rp /Users/brad/Scripts/python/metis/metisgh/projects/kojak/nlsy79/data/ brad@$(echo $(ipaws) | sed -e 's/^"//' -e 's/"$//'):/home/brad/nlsy79/data/
scp -rp /Users/brad/Scripts/python/metis/metisgh/projects/kojak/nlsy79/marstat/ brad@$(echo $(ipaws) | sed -e 's/^"//' -e 's/"$//'):/home/brad/nlsy79/marstat/
