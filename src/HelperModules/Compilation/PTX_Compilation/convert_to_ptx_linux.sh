#! /bin/bash
#
#
# Key Requirements: 
# 1. Script parses command-line arguments
#  Cases:
#    Case: No arguments are supplied 
#    Expected Response: Print output (see hw guide) to standard error stream (2) and exit with 1
#
#    Case: User passes -h flag  [DONE]
#    Expected Response: Print output to standard out stream (1) and exit with 0
#
#    Case: Unexpected flag OR Combination of Unexpected and Expected Flags
#    Expected Response: Display error message ; Print correct usage (see hw guide)
#
#    Case: Multiple Erroneous/Unexpected flags
#    Expected Response: Display error message but with first flag reported only ; Print correct usage
#
#    Case: Multiple Valid & Distinct flags
#    Expected Response: Display error message (see hw guide); Print correct usage
#
#    Case: Multiple Valid & Distinct flags AND list of files
#    Expected Response: Display error message (see hw guide); Print correct usage
#
# SOURCES: 
#
#===================================================================PROGRAM==============================================================


#GLOBAL VARIABLES

help_flag=0 #indicates presence of -h help flag

list_flag=0 #indicates presence of -l list flag

purge_flag=0 #indicates presence of -p purge flag

total_valid_flags=0 #indicates total number of valid (-h, -l or -p) flags

file_count=0 #counts number of arguments that are files (and are not any of the above valid flags)


scriptName=$(basename "$0")

USAGE=$(cat <<- EOF
Usage: $scriptName [-hlp] [list of files]
   -h: Display help.
   -l: List junked files.
   -p: Purge all files.
   [list of files] with no other arguments to junk those files.
EOF
)

directory_count=0
symlink_count=0

directory_flag=0
symlink_flag=0

readonly target_directory="$HOME/.junk"


#FUNCTION: recurse_dir 
#
#Description: Function to recurse file system. 
#Implementation is from search.sh file created on Jan. 26, 2023 from COMS W3157 lecture.

# Function to recurse the file system.

recurse_dir() {
    # "$1"/* matches all files except hidden files.
    # "$1"/.[!.]* matches hidden files, but not .. which would lead to
    # infinite recursion.
    for file in "$1"/.[!.]* "$1"/*; do #match files with a . but not a second dot ; "$1"/* means everything else
#        echo "The current file is $file"
        # -h tests if a file is a symlink.
        if [ -h "$file" ]; then 
            # readlink prints the location to which the symlink points.
            echo "symlink  : $file -> $(readlink "$file")" #readlink tells where symlink points to & path
            (( ++symlink_count ))
        fi
        # -d tests if a file is a directory.
        if [ -d "$file" ]; then
            if [ "$directory_flag" -eq 1 ]; then
                echo "directory: $file"
                (( ++directory_count ))
            fi
            recurse_dir "$file" #recurse to find symbolic links if present
        fi
	#echo "File count is $file_count"
    done
}


#FUNCTION: parseJunkArgs()
#
#Description: 
#  Review flag count:
#     1) Check if there are no arguments
#     2) Otherwise, call getopts to parse arguments and increment total valid flags
#     3) Return error for first unknown flag provided

parseJunkArgs() {
  while getopts ":hlp" option; do
     case "$option" in
	h)
	   help_flag=1
	   ((total_valid_flags++))
	   ;;
        l)
	    list_flag=1
	    ((total_valid_flags++))
	    ;;
	p)
            purge_flag=1
	    ((total_valid_flags++))
	    ;;
	?)
            printf "Error: Unknown option '-%s.\n" "$OPTARG" >&2
	       echo "$USAGE"
	       exit 1
	    ;;
      esac
   done

   shift "$((OPTIND-1))"


   if [ $# -gt 1 ]; then
      echo "Error: Too many arguments." >&2
      exit 1
   elif [ $# -eq 0 ]; then
      recurse_dir .
   else
      recurse_dir "$1"
   fi

   # Print the counts discovered during the search.
    if [ "$symlink_flag" -eq 1 ]; then
        if [ "$symlink_count" -eq 1 ]; then
            echo "1 symlink found."
        else
            echo "$symlink_count symlinks found."
        fi
    fi
    
    if [ "$directory_flag" -eq 1 ]; then
        if [ "$directory_count" -eq 1 ]; then
            echo "1 directory found."
        else
            echo "$directory_count directories found."
        fi
    fi
}


 #FUNCTION: checkForJunkDir ()
 #
 #Description: 
 #   1) Check for presence of .junk directory
 #   2) Create .junk directory if not available
 
 checkForJunkDir () {
    ls -d "$target_directory" 1> junkDirPresent 2> junkDirAbsent
    if grep -q "No such file or directory" junkDirAbsent; then
       mkdir "$target_directory"
       rm junkDirAbsent
    elif grep -q "No such file or directory" junkDirPresent; then
       mkdir "$target_directory"
       rm junkDirPresent
    else
       rm junkDirAbsent
       rm junkDirPresent
    fi
}


#FUNCTION: evaluateFlags ()
#
#Description:
#  1) Checks for presence of each flag and total number of valid flags
#  2) Evaluates conditional statements and takes certain actions depending on each flag

evaluateFlags () {
     if [ $help_flag -eq 1 ] && [ $total_valid_flags -eq 1 ]; then
       # printf 'Number of valid flags: %s'"$total_valid_flags\n"
        echo "$USAGE"
        exit 0
     fi
     if [ $list_flag -eq 1 ] && [ $total_valid_flags -eq 1 ]; then
         printf "Checking for presence of junk directory\n"
	 checkForJunkDir
	 printf "Junk directory exists. Now listing files.\n"
         cd "$target_directory" || exit
	 ls -lAF
      fi
      if [ $purge_flag -eq 1 ] && [ $total_valid_flags -eq 1 ]; then
	   printf "Checking for presence of junk directory ...\n\n"
	   checkForJunkDir
	   printf "Junk directory exits. Now purging files..\n"
	   find "$target_directory" -mindepth 1 -delete
      fi
      if [ $total_valid_flags -gt 1 ]; then
	      printf "Error: Too many options enabled.\n" >&2
	      echo "$USAGE"
	      exit 1
      fi
}

#FUNCTION: convertToPtx_traditional()
#
#Description: 

#FUNCTION: main()
#
#Description:
# 1) Parse command-line arguments using parseJunkArgs() function
# 2) Evaluate flags and either print usage statement, error messages, list files in .junk directory or purge files in .junk directory

main() {
   parseJunkArgs "$@"
   evaluateFlags
}

if [ $# -eq 0 ]; then
   printf "Error: No arguments provided.\n" >&2
   echo "$USAGE"
   exit 1
else
    main "$@"
fi

#Confirmation of successful run
exit 0

