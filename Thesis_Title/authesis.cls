% Identification
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{authesis}[2026/03/27 Ver 0.1]

% Initialization
\LoadClass{book}

% Options
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions \relax

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Class code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%
% Define variables
%%%%%%%%%%%%%%%%%%%
\newcommand{\department}[1]{\def\@department{#1}}
\newcommand{\college}[1]{\def\@college{#1}}
\newcommand{\place}[1]{\def\@place{#1}}
\newcommand{\supervisor}[1]{\def\@supervisor{#1}}
\newcommand{\supervisortitle}[1]{\def\@supervisortitle{#1}}
\newcommand{\supervisordepartment}[1]{\def\@supervisordepartment{#1}}
\newcommand{\supervisorcollege}[1]{\def\@supervisorcollege{#1}}
\newcommand{\supervisorplace}[1]{\def\@supervisorplace{#1}}
\newcommand{\dateofsubmission}[1]{\def\@dateofsubmission{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Student names display in title page, Bonafide and acknowledgement
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\@students}{}   % vertical list
\newcommand{\@studentspara}{}        % For paragraph display
%\RequirePackage{varwidth}
%\newcommand{\student}[2]{%
%  \g@addto@macro\@students{%
%    \begin{varwidth}[t]{0.7\textwidth}
%      \raggedright\MakeUppercase{#1}
%    \end{varwidth}
%    & #2\tabularnewline
%  }%
%}

\newcount\studentcount
\newcount\nauthors

\renewcommand{\author}[1]{%
  \gdef\@students{}%
  \gdef\@studentspara{}%
  \studentcount=0
  \nauthors=0
  #1%
}

\newcommand{\student}[2]{%
  % Vertical list
  \g@addto@macro\@students{%
    \MakeUppercase{#1}\par
  }%

  \advance\studentcount by 1

  % Store entries with a marker
  \ifnum\studentcount=1
    \g@addto@macro\@studentspara{#1 (#2)}%
  \else
    \g@addto@macro\@studentspara{|#1 (#2)}%
  \fi

  % --- NEW: Bottom-aligned macros ---
  \advance\nauthors by 1
  \ifnum\nauthors=1 \expandafter\gdef\csname authorone\endcsname{\textbf{\MakeUppercase{#1}}}\fi
  \ifnum\nauthors=2 \expandafter\gdef\csname authortwo\endcsname{\textbf{\MakeUppercase{#1}}}\fi
  \ifnum\nauthors=3 \expandafter\gdef\csname authorthree\endcsname{\textbf{\MakeUppercase{#1}}}\fi
  \ifnum\nauthors=4 \expandafter\gdef\csname authorfour\endcsname{\textbf{\MakeUppercase{#1}}}\fi
  \ifnum\nauthors=5 \expandafter\gdef\csname authorfive\endcsname{\textbf{\MakeUppercase{#1}}}\fi
}

%%%%%%%%%%%%%%%%%%%%%%%%
% Student names in Bonafide
%%%%%%%%%%%%%%%%%%%%%%%%
% ---------- Print paragraph inline with commas and 'and' ----------
\newcommand{\printstudentwithreg}{%
  \begingroup
  \def\result{}%
  \def\process##1|##2\relax{%
    \def\temp{##2}%
    \ifx\temp\empty
      \ifx\result\empty
        \def\result{##1}%
      \else
        \edef\result{\result\ and ##1}%
      \fi
    \else
      \ifx\result\empty
        \def\result{##1}%
      \else
        \edef\result{\result, ##1}%
      \fi
      \process##2\relax
    \fi
  }%
  \expandafter\process\@studentspara|\relax
  \result
  \endgroup
}


%%%%%%%%%%%%%%%%%%%%%%%%
% Student names in Acknowledgement (bottom-aligned)
%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\printstudentsbottom}{%
  \ifnum\nauthors=1
    \begin{center}
      \authorone
    \end{center}
  \fi

  \ifnum\nauthors=2
    \begin{center}
      \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} c c @{}}
        \authorone & \authortwo
      \end{tabular*}
    \end{center}
  \fi

  \ifnum\nauthors=3
    \begin{center}
      \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} c c c @{}}
        \authorone & \authortwo & \authorthree
      \end{tabular*}
    \end{center}
  \fi

  \ifnum\nauthors=4
    \begin{center}
      \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} c c c @{}}
        \authorone & \authortwo & \authorthree
      \end{tabular*}
      
      \vspace{1em}
      \authorfour
    \end{center}
  \fi

  \ifnum\nauthors=5
    \begin{center}
      \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} c c c @{}}
        \authorone & \authortwo & \authorthree
      \end{tabular*}
      
      \vspace{1em}
      \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} c c @{}}
        \authorfour & \authorfive
      \end{tabular*}
    \end{center}
  \fi
}















%%%%%%%%%%%%%%%%%%
% Page dimensions
%%%%%%%%%%%%%%%%%%
\RequirePackage[top=35mm,bottom=30mm,left=40mm,right=25mm,footskip=20mm,headsep=11mm, twoside=false]{geometry}
%\RequirePackage[showframe=true,top=35mm,bottom=30mm,left=40mm,right=25mm,footskip=20mm,headsep=11mm]{geometry}

%%%%%%%%%%%%%%%%%
% Font selection
%%%%%%%%%%%%%%%%%
\RequirePackage{iftex}
\ifPDFTeX
	\RequirePackage[T1]{fontenc}
	\RequirePackage[utf8]{inputenc}
	\RequirePackage{newtxtext,newtxmath}
\else
	\RequirePackage{fontspec}
	\IfFontExistsTF{Times New Roman}{
		\setmainfont[Ligatures=TeX]{Times New Roman}
	}{
		\setmainfont[Ligatures=TeX]{TeX Gyre Termes}
	}
\fi
% note that default baselineskip is 1.15 x fontsize
\def\normalsize{\fontsize{13pt}{14.95pt}\selectfont}
% 1.5 line spacing
\renewcommand{\baselinestretch}{1.5}

%%%%%%%%%%%%%%%%%%%%%
% Paragraph settings
%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{indentfirst}
\parindent 20mm
\parskip 1em

%%%%%%%%%%%%%%%%%
% Citation style
%%%%%%%%%%%%%%%%%
\RequirePackage[numbers,sort&compress]{natbib}


%%%%%%%%%%%%%
% Title page
%%%%%%%%%%%%%
\newcommand{\aumaketitle}{
	\begin{titlepage}
        \thispagestyle{empty}
		\parskip 0cm
		\centering
        
		\fontsize{18pt}{20.7pt}\selectfont \textbf{\expandafter\uppercase\expandafter{\@title}}
        
		\vskip 0cm plus2fill
		\fontsize{14pt}{16.1pt}\selectfont \textbf{A PROJECT REPORT}
        
		\vskip 0cm plus2fill
		\fontsize{14pt}{16.1pt}\selectfont \textbf{\textit{Submitted by}}
        
		\vskip 0cm plus1fill        
        %%%%%%%%%%%%%% BEGIN BE THESIS %%%%%%%%%
        \fontsize{16pt}{18.4pt}\selectfont
        \textbf{%
%            \begin{tabular}{@{}l p{0.2\textwidth}@{}}
%                \@students
%            \end{tabular}
                \@students
        }
        %%%%%%%%%%%%%% END BE THESIS %%%%%%%%%
		\vskip 0cm plus2.5fill
		\fontsize{14pt}{16.1pt}\selectfont 
			\textbf{\textit{in partial fulfillment of the requirements for the degree of}}
		\vskip 0cm plus1.5fill
		\fontsize{16pt}{18.4pt}\selectfont 

        %%%%%%%%%%%%%% BEGIN BE THESIS %%%%%%%%%
        \textbf{BACHELOR OF ENGINEERING}

        \vskip 0cm plus1.5fill
		\fontsize{14pt}{16.1pt}\selectfont 
			\textbf{IN}
        
		\vskip 0cm plus1.5fill
		\fontsize{16pt}{18.4pt}\selectfont COMPUTER SCIENCE AND ENGINEERING
        %%%%%%%%%%%%%% END BE THESIS %%%%%%%%%
        
        \vskip 0cm plus2.5fill
		\includegraphics[width=3cm]{emblem.png}
        
        %%%%%%%%%%%%%% BEGIN BE THESIS %%%%%%%%%
		\vskip 0cm plus1fill
		\fontsize{14pt}{16.1pt}\selectfont 
			\textbf{COLLEGE OF ENGINEERING GUINDY}
        %%%%%%%%%%%%%% END BE THESIS %%%%%%%%%
        
		\vskip 0cm plus1fill
		\fontsize{16pt}{18.4pt}\selectfont \textbf{ANNA UNIVERSITY : CHENNAI 600 025}
        
   
		\vskip 0cm plus1fill
		\fontsize{14pt}{16.1pt}\selectfont \textbf{\expandafter\uppercase\expandafter{\@dateofsubmission}}
        
	\end{titlepage}
}

% Certificate
\newcommand{\aumakecertificate}{
		{
        \thispagestyle{empty}
		\begin{center} 
			\fontsize{16pt}{18.4pt}\selectfont 
			\textbf{ANNA UNIVERSITY : CHENNAI 600 025}

			\textbf{BONAFIDE CERTIFICATE}
		\end{center}
		
		\parindent 0pt
		Certified that this project report \textbf{``\MakeUppercase{\@title}''}
		is the bonafide work of \textbf{\printstudentwithreg} who carried out the project work under my supervision.

		\vskip 5em

		\noindent\parbox[t]{.45\linewidth}{
			\textbf{
            SIGNATURE\\
            Dr. \MakeUppercase{V. Mary Anita Rajam}\\
            HEAD OF THE DEPARTMENT\\
            }
            Department of Computer Science and Engineering,\\
            \\
            Anna University,\\
            Chennai 600025.
		}
        \hspace{0.1\linewidth}% <-- add horizontal space
		\parbox[t]{.45\linewidth}{
			\textbf{
            SIGNATURE\\
            Dr. \MakeUppercase{\@supervisor} \\
			SUPERVISOR\\
            }
			\@supervisortitle, \\
			\@supervisordepartment, \\
			\@supervisorcollege, \\
			\@supervisorplace.
		}

		}
		\clearpage
}

%%%%%%%%%%%%%%%%%%
% Acknowledgement
%%%%%%%%%%%%%%%%%%
\newcommand{\aumakeacknowledgement}{
        \thispagestyle{empty}
		{
		\begin{center} 
			\fontsize{16pt}{18.4pt}\selectfont 
			\textbf{ACKNOWLEDGEMENT}
		\end{center}
		
		We express our deep gratitude to our guide, \textbf{Dr. \@supervisor}, \@supervisortitle, \@supervisordepartment, for guiding us through every phase of the project. We appreciate her thoroughness, tolerance and ability to share her knowledge with us. We would also like to thank her for her kind support and for providing necessary facilities to carry out the work.  

We are extremely grateful to \textbf{Dr. V. Mary Anita Rajam}, Professor \& Head of the Department, Department of Computer Science and Engineering, Anna University, Chennai – 25, for extending the facilities of the Department towards our project and for her unstinting support.  

We express our thanks to the panel of reviewers \textbf{Dr. P. Geetha}, Professor, Department of Computer Science and Engineering and \textbf{Dr. S. Sudha}, Professor, Department of Computer Science and Engineering, for their valuable suggestions and critical reviews throughout the course of our project. 

We express our thanks to all other teaching and non-teaching staff who helped us in one way or other for the successful completion of the project. We would also like to thank our parents, family and friends for their indirect contribution in the successful completion of this project.
\vfill
 \printstudentsbottom
        }
		\clearpage
}


%%%%%%%%%%%
% Abstract
%%%%%%%%%%%
\newenvironment{abstract}{
	% begin environment commands
	\hrule height 0pt % to ensure the following vskip at beginning of page works
	\vskip 5mm
	{ \fontsize{16pt}{18.4pt}\selectfont \begin{center} \textbf{ABSTRACT} \end{center} }
	\addcontentsline{toc}{chapter}{\hskip \cftchapnumwidth ABSTRACT}
}{
	% end environment commands
	\clearpage
}


%%%%%%%%%%%%%%%%%%%%
% Table of contents
%%%%%%%%%%%%%%%%%%%%
\RequirePackage{hyperref}
\newcommand{\autableofcontents}{
	\toctrue
	\tableofcontents
	\tocfalse
}

%%%%%%%%%%%%%%%%%
% List of tables
%%%%%%%%%%%%%%%%%
\newcommand{\aulistoftables}{
	\clearpage
	\addcontentsline{toc}{chapter}{\hskip \cftchapnumwidth LIST OF TABLES}
	\lottrue
	\listoftables
	\lotfalse
}


%%%%%%%%%%%%%%%%%%
% List of figures
%%%%%%%%%%%%%%%%%%
\newcommand{\aulistoffigures}{
	\clearpage
	\addcontentsline{toc}{chapter}{\hskip \cftchapnumwidth LIST OF FIGURES}
	\loftrue
	\listoffigures
	\loffalse
}

%%%%%%%%%%%%%%%%%%
% List of symbols
%%%%%%%%%%%%%%%%%%
\newenvironment{listofsymbols}{
	\clearpage
	% begin environment commands
	\parindent 0pt
	\parskip 0pt
	{ \fontsize{14pt}{16.1pt}\selectfont \begin{center} \textbf{LIST OF SYMBOLS} \end{center} }
	\vskip 2em
	\addcontentsline{toc}{chapter}{\hskip \cftchapnumwidth LIST OF SYMBOLS}
}{
	% end environment commands
}
\newcommand{\entry}[2]{
	\parbox[t]{30mm}{#1}
	\parbox[t]{10mm}{-}
	\parbox[t]{105mm}{#2}
	\par
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Table of contents - styles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{tocloft}
\RequirePackage{etoolbox}
\setcounter{secnumdepth}{6}
\setcounter{tocdepth}{7}
% set page number width and right margin of title
\cftsetpnumwidth{3em}
\cftsetrmarg{4em}
% remove the dots
\renewcommand{\cftdot}{}
% table of contents (toc) title
\setlength{\cftbeforetoctitleskip}{0pt}
\setlength{\cftaftertoctitleskip}{0pt}
\renewcommand{\cfttoctitlefont}{\fontsize{14pt}{16.1pt}\selectfont \bf \hfill}
\renewcommand{\cftaftertoctitle}{
	{\hfill} \vskip 2em {\fontsize{12pt}{13.8pt}\selectfont \bf CHAPTER NO. \hfill TITLE \hfill PAGE NO.}
}
\renewcommand{\contentsname}{TABLE OF CONTENTS}

%% Repeat TOC, LOF and LOT headers on all pages.
\RequirePackage{atbegshi}
\newif\iftoc
\newif\iflot
\newif\iflof
\AtBeginShipout{
	\iftoc
		\fontsize{12pt}{13.8pt}\selectfont
		\bf CHAPTER NO. \hfill TITLE \hfill PAGE NO.
		\vskip 1em
	\else
		\iflot
			\fontsize{12pt}{13.8pt}\selectfont
			\bf TABLE NO. \hfill TITLE \hfill PAGE NO.
			\vskip 1em
		\fi
		\iflof
			\fontsize{12pt}{13.8pt}\selectfont
			\bf FIGURE NO. \hfill TITLE \hfill PAGE NO.
			\vskip 1em
		\fi
	\fi
}

% toc chapter entry
\setlength{\cftchapindent}{0pt}
\patchcmd{\l@chapter}
	{\cftchapfont #1}		% search pattern
	{\cftchapfont {#1}}	% replace by
	{}									% success
	{}									% failure
\renewcommand{\cftchapfont}{\bf\uppercase}
\renewcommand{\cftchappagefont}{\bf}
\setlength{\cftchapnumwidth}{1.5cm}
\renewcommand{\cftchappresnum}{\hfill}
\renewcommand{\cftchapaftersnum}{\hfill}
% toc section entry
\setlength{\cftsecindent}{\cftchapnumwidth}
\patchcmd{\l@section}
	{\cftsecfont #1}		% search pattern
	{\cftsecfont {#1}}		% replace by
	{}									% success
	{}									% failure
\renewcommand{\cftsecfont}{\uppercase}
\setlength{\cftsecnumwidth}{1.2cm}
% toc subsection entry
\setlength{\cftsubsecindent}{\cftsecindent}
\addtolength{\cftsubsecindent}{\cftsecnumwidth}
\setlength{\cftsubsecnumwidth}{1.4cm}
% toc subsubsection entry
\setlength{\cftsubsubsecindent}{\cftsubsecindent}
\addtolength{\cftsubsubsecindent}{\cftsubsecnumwidth}
\setlength{\cftsubsubsecnumwidth}{1.6cm}
% toc paragraph entry
% \cftsetindents{<entry>}{<indent>}{<numwidth>}
\cftsetindents{paragraph}{2.24in}{1.8cm}
% toc subparagraph entry
% \cftsetindents{<entry>}{<indent>}{<numwidth>}
\cftsetindents{subparagraph}{2.95in}{2cm}
% list of figures (lof) title
\setlength{\cftbeforeloftitleskip}{0pt}
\setlength{\cftafterloftitleskip}{0pt}
\renewcommand{\cftloftitlefont}{\bf \hfill}
\renewcommand{\cftafterloftitle}{
	{\hfill} \vskip 2em {\fontsize{12pt}{13.8pt}\selectfont \bf FIGURE NO. \hfill TITLE \hfill PAGE NO.}
}
\renewcommand{\listfigurename}{LIST OF FIGURES}
% lof entry
\setlength{\cftfigindent}{0pt}
\setlength{\cftfignumwidth}{3cm}
\renewcommand{\cftfigpresnum}{\hfill}
\renewcommand{\cftfigaftersnum}{\hfill}
% list of tables (lot) title
\setlength{\cftbeforelottitleskip}{0pt}
\setlength{\cftafterlottitleskip}{0pt}
\renewcommand{\cftlottitlefont}{\bf \hfill}
\renewcommand{\cftafterlottitle}{
	{\hfill} \vskip 2em {\fontsize{12pt}{13.8pt}\selectfont \bf TABLE NO. \hfill TITLE \hfill PAGE NO.}
}
\renewcommand{\listtablename}{LIST OF TABLES}
% lot entry
\setlength{\cfttabindent}{0pt}
\setlength{\cfttabnumwidth}{3cm}
\renewcommand{\cfttabpresnum}{\hfill}
\renewcommand{\cfttabaftersnum}{\hfill}


%%%%%%%%%%%%%%%%%
% Page numbering
%%%%%%%%%%%%%%%%%
% page style for preliminary pages like title, certificate, abstract, etc
\newcommand{\ps@authesispreliminary}{
	\renewcommand{\@oddhead}{}
	\renewcommand{\@evenhead}{}
    \renewcommand{\@oddfoot}{\hfill \thepage \hfill}   % centered page number
    \renewcommand{\@evenfoot}{\hfill \thepage \hfill}  % same for even pages
    \pagenumbering{roman}
}
% page style for chapter pages
\newcommand{\ps@authesischapters}{
    \renewcommand{\@oddhead}{}   % no header
    \renewcommand{\@evenhead}{}  % no header

    \renewcommand{\@oddfoot}{\hfill \thepage \hfill}   % centered page number
    \renewcommand{\@evenfoot}{\hfill \thepage \hfill}  % same for even pages

    \pagenumbering{arabic}
}
% redefine plain so that preliminary pages and first pages
% of chapters which use page style 'plain' by default
% are also changed
\renewcommand{\ps@plain}{
	\renewcommand{\@oddhead}{}
	\renewcommand{\@evenhead}{}
}

% preliminary pages like title, certificate, abstract, etc are one sided
\newcommand{\authesisfrontmatter}{
	\newgeometry{top=35mm,bottom=30mm,left=40mm,right=25mm,footskip=20mm,headsep=11mm,twoside=false}
	\pagestyle{authesispreliminary}
}
% restore to two sided pages for chapters
\newcommand{\authesischapters}{
	\restoregeometry
	\pagestyle{authesischapters}
}

%%%%%%%%%%%%%%%
% Title styles
%%%%%%%%%%%%%%%
\RequirePackage{titlesec}

% \titleformat{command}[shape]{title_format}{label}{hspace_label_title}{before_title}{after_title}
% \titlespacing{command}{left_margin}{vspace_before}{vspace_after}[right_margin]
\titleformat{\chapter}[display]
	{\fontsize{14pt}{16.1pt}\selectfont \bf \parskip 0pt \filcenter}
	{CHAPTER \thechapter}{2em}{\MakeUppercase}{}		
\titlespacing{\chapter}{0pt}{0pt}{2em}

\titleformat{\section}
	{\bf}
	{\thesection}{3.15em}{\MakeUppercase}{}	
\titlespacing{\section}{0pt}{1.1em}{0.6em}

\titleformat{\subsection}
	{\bf}
	{\thesubsection}{2.4em}{}{}	
\titlespacing{\subsection}{0pt}{0.9em}{0.5em}

\titleformat{\subsubsection}
	{\bf}
	{\thesubsubsection}{1.65em}{}{}	
\titlespacing{\subsubsection}{0pt}{0.8em}{0.4em}

\titleformat{\paragraph}
	{\bf}
	{\theparagraph}{.9em}{}{}
\titlespacing{\paragraph}{0pt}{0.6em}{0.3em}

\titleformat{\subparagraph}
	{\bf}
	{\thesubparagraph}{0.15em}{}{}
\titlespacing{\subparagraph}{0pt}{0.5em}{0.2em}

%%%%%%%%%%%%%%%
% Environments for Algorithm, Figure, Table, Itemize and Enumerate
%%%%%%%%%%%%%%%

% one line spacing for algorithms and longtables
\RequirePackage{setspace}
\AtBeginEnvironment{algorithm}{\setstretch{1}}
%\AtBeginEnvironment{longtable}{\setstretch{1}}

% Caption style for table and figure
\RequirePackage[format=hang,labelfont=bf,textfont=bf,font=normalsize]{caption}
\captionsetup{
    width=0.9\textwidth,
    justification=centering,
	singlelinecheck=false,
	skip=0.75em,
	position=bottom
}
\setlength{\textfloatsep}{1.2em}
\setlength{\floatsep}{1em}
\setlength{\intextsep}{1em}

% Caption position helpers: use \tablecaptiontop or \tablecaptionbottom where needed.
\newcommand{\tablecaptiontop}{\captionsetup[table]{position=top}}
\newcommand{\tablecaptionbottom}{\captionsetup[table]{position=bottom}}

% List (itemize and enumerate) indentation
\RequirePackage[shortlabels]{enumitem}
%\RequirePackage[nameinlink]{cleveref}	% display only number if \ref is used and display word and number if \cref is used
\setlist{leftmargin=5.8em,labelsep=1em,topsep=0.6em,itemsep=0.4em,parsep=0pt,partopsep=0pt}
\renewcommand{\labelitemi}{\textbullet}



%%%%%%%%%%%%%%%%%%%%%%
% Bibliography styles
%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\bibname}{REFERENCES}
\newcommand{\references}[1]{
	\clearpage
	\phantomsection
	\addcontentsline{toc}{chapter}{\hskip \cftchapnumwidth REFERENCES}	
	\bibliography{#1}
}

%%%%%%%%%%%%%%%%%%%%%%
% List of Publications
%%%%%%%%%%%%%%%%%%%%%%
\newcounter{lpubcounter}  
\newenvironment{lpubenum}{
	% begin environment commands

	% Heading style
	\chapter*{LIST OF PUBLICATIONS}

	% ToC Entry
	\addcontentsline{toc}{chapter}{\hskip \cftchapnumwidth LIST OF PUBLICATIONS}	

	\noindent\parskip 0.5em\textbf{International Journals}
	% List style
 	\begin{list}
 		{\arabic{lpubcounter}.}
 		{
 			\usecounter{lpubcounter} 
 			\labelsep=1.19em 
 			\labelwidth=1.19em 
 			\leftmargin=1.8em 
 			\itemindent=0em
 		}
 }{
	% end environment commands
 	\end{list}
 }

%%%%%%%%%%%%%%%%%%%%%%
% Appendix
%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\appendix}{

	% Reset chapter numbering
	\setcounter{chapter}{0}
	\setcounter{section}{0}

	% Ensure appendix anchors are distinct from main chapters.
	\renewcommand{\theHchapter}{appendix.\arabic{chapter}}
	\renewcommand{\theHsection}{appendix.\arabic{chapter}.\arabic{section}}

	% ToC entry
	\clearpage
	\phantomsection
	\addcontentsline{toc}{chapter}{\hskip \cftchapnumwidth APPENDIX}	
	
	% Chapter title - Replace the word "Chapter" with "Appendix"
	% \titleformat{command}[shape]{title_format}{label}{hspace_label_title}{before_title}{after_title}
	% \titlespacing{command}{left_margin}{vspace_before}{vspace_after}[right_margin]
	\titleformat{\chapter}[display]
		{\fontsize{14pt}{16.1pt}\selectfont \bf \parskip 0pt \filcenter}
		{APPENDIX \thechapter}{2em}{\MakeUppercase}{}
	\titlespacing{\chapter}{0pt}{0pt}{2em}

	% Figure caption
	\renewcommand{\thefigure}{A \thechapter.\arabic{figure}}
	
	% Table caption
	\renewcommand{\thetable}{A \thechapter.\arabic{table}}
}

% prevent hyphenated lines at page endings
\brokenpenalty10000\relax

% keep widows/orphans discouraged, but allow TeX to avoid large page-bottom gaps
\clubpenalty3000
\widowpenalty3000
\displaywidowpenalty3000
