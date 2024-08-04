for dir in */; do
    rm -f "$dir"/*
    touch "$dir/init.py"
done
