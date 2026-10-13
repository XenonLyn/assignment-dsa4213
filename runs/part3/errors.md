# Representative test errors

## best LM representation (lstm_glove/mean) wrong, TF-IDF right (621 docs)

| true | lstm_glove/mean | tfidf | <unk> share | text |
|---|---|---|---|---|
| Sci/Tech | Business | Sci/Tech | 0.02 | Dell May Soon Unveil More Consumer Goods -Analyst (Reuters) Reuters - Dell Inc. (DELL.O), the world's\largest PC maker, could announce an expanded selection of its\consumer electronics line in the next several weeks, a r |
| Sci/Tech | Business | Sci/Tech | 0.07 | FCC expected to keep states off VoIP's back Sources expect that on Tuesday, the FCC will exempt more Net phone calls from state telephone rules and taxes, even as the cable industry tries to grab VoIP's coattails. |
| World | Sci/Tech | World | 0.07 | FDA OKs Scientist Publishing Vioxx Data (AP) AP - The Food and Drug Administration has given a whistle-blower scientist permission to publish data indicating that as many as 139,000 people had heart attacks that may be l |
| Business | Sci/Tech | Business | 0.00 | AOL to cut 700 jobs America Online Inc. (AOL) plans to lay off 700 employees, about 5 percent of its US workforce, by the end of the year, several news organizations reported Tuesday. |
| Sports | Sci/Tech | Sports | 0.03 | ESPN.com news services Six months ago, Scottie Pippen issued a  quot;This is probably it for me quot; declaration, that last season was looking more and more like his last in an NBA uniform. |
| Business | World | Business | 0.08 | Parmalat Investors Flock to Court Seeking Damages (Update2) Hundreds of Italians who lost savings in the collapse of food company Parmalat Finanziaria SpA flocked to Milan #39;s courthouse, seeking to recover damages as  |
| World | Business | World | 0.08 | Mich. Rep. to Head Intelligence Panel (AP) AP - Republican Rep. Peter Hoekstra of Michigan was picked Wednesday to head the House Intelligence Committee amid a heated election-year debate over how to carry out a major ov |
| Sports | Business | Sports | 0.06 | Pot charge dropped against Anthony In a case that his lawyer said  quot;has received more prosecutorial scrutiny than any petty offense in Denver history, quot; Nuggets forward Carmelo Anthony saw the marijuana charge he |

## best LM representation (lstm_glove/mean) right, TF-IDF wrong (221 docs)

| true | lstm_glove/mean | tfidf | <unk> share | text |
|---|---|---|---|---|
| World | World | Sports | 0.00 | Canada defends submarine fleet Canada has defended its decision to buy second-hand submarines after a crewman died from injuries sustained on one of the vessels that had broken down. |
| Sports | Sports | World | 0.03 | SEC's Gaston Admits Official Missed Call (AP) AP - The Southeastern Conference's supervisor of officials said an LSU player should have been called for pass interference on a pivotal interception against Alabama. |
| Sci/Tech | Sci/Tech | Business | 0.08 | Sony looks under the Christmas tree Americans will go gadget shopping this holiday season even if oil prices go up, Sony execs say. |
| Business | Business | Sci/Tech | 0.04 | Level 3 Acquires Sprint #39;s Wholesale Dial-Up Internet Business Level 3 today announced that it has purchased Sprint #39;s wholesale dial-up Internet access business for \$34 million in cash. Sprint is one of the large |
| Sports | Sports | World | 0.13 | Italian Wins Marathon... US Finishes Second Italian Stefano Baldini has won the men #39;s marathon in a time of 2:10:54. Naturalized American Meb Keflezighi was a surprise runnerup with Brazil #39;s Vanderlei Lima finish |
| Business | Business | Sci/Tech | 0.03 | Federal CISOs Rank Patch Management As Biggest Obstacle Survey by Intelligent Decisions indicates that patch management leaves less time for chief information security officers to work on improving overall security. |
| Sci/Tech | Sci/Tech | Business | 0.21 | TSMC, Freescale expect initial production of 65nm SOI in 4Q 2005 Taiwan Semiconductor Manufacturing Company (TSMC) and Freescale Semiconductor expect to begin initial production of a high-speed 65nm silicon-on-insulator  |
| Sci/Tech | Sci/Tech | Business | 0.08 | Matsushita Unveils DVD Recorders, Eyes Higher Share  TOKYO (Reuters) - Panasonic brand products maker Matsushita  Electric Industrial unveiled five new DVD recorders on  Wednesday and said it was aiming to boost its shar |

## every LM representation wrong (224 docs)

| true | lstm_glove/mean | tfidf | <unk> share | text |
|---|---|---|---|---|
| Business | Sci/Tech | Sci/Tech | 0.12 | Sony, IBM, Toshiba Give Details of 'Cell' Chip (Reuters) Reuters - IBM , Sony Corp. (6758.T) and\Toshiba Corp. (6502.T) on Monday revealed their plans for the\powerful new "Cell" processor the three are jointly producing |
| Business | World | Business | 0.00 | CAO executives hand over passports Executives at the collapsed China Aviation Oil Singapore have voluntarily handed over their passports to Singapore #39;s police, a spokesman said Tuesday. |
| Business | Sci/Tech | Sci/Tech | 0.05 | Retailers Looking to Move Plasma TV's (AP) AP - Hanging stockings by the chimney with care? Retailers hope that St. Nicholas soon will be there  #151; to hang a 42-inch plasma-screen TV. |
| Business | Sports | Sports | 0.02 | Special to ESPN.com It #39;s the age old question:  quot;What do you give to the man who #39;s been everything? quot;. Only time will tell whether Phil Knight #39;s retirement will be as long-lived as so many players he  |
| World | Sports | Sports | 0.05 | Warne takes six but India establish handy lead (Reuters) Reuters - World test wicket record holder Shane Warne grabbed six wickets as India established a handy 141-run first innings lead in the second test on Saturday. |
| Business | Sci/Tech | Sci/Tech | 0.06 | Research Is Definitely in Motion The Blackberry wireless device maker is straining to exceed expectations. |
| Sports | Business | Sports | 0.03 | Judge declines to dismiss BALCO case A federal judge said Wednesday that she would not immediately dismiss charges against four men accused of distributing steroids to top athletes. |
| World | Sports | World | 0.05 | Pakistan down India to ensure top six finish (AFP) AFP - Pakistan defeated arch-rivals India 3-0 here to ensure they stand among the top six in the Olympic men's field hockey competition. |
