#Convert geometry (final) in all Gaussian .out files in current folder to .gjf file by Multiwfn, internal coordinate is used
#!/bin/bash
icc=0
nfile=`ls *.sdf|wc -l`
for inf in *.sdf
do
((icc++))
echo Converting ${inf} to ${inf//sdf/gjf} ... \($icc of $nfile\)
Multiwfn ${inf} << EOF > /dev/null
geomparm
geomparm ${inf//sdf/txt}
q
EOF
done
