#!/bin/bash
# Monitor benchmark progress and output updates
LAST_COUNT=0
while true; do
    # Check if benchmark still running
    if ! ps -p 57200 > /dev/null 2>&1; then
        echo "BENCHMARK_FINISHED"
        break
    fi
    
    # Count DB files (each = 1 completed sample)
    CURRENT_COUNT=$(ls /tmp/reasoning-memory/benchmark_dbs/*.db 2>/dev/null | wc -l | tr -d ' ')
    
    if [ "$CURRENT_COUNT" != "$LAST_COUNT" ]; then
        echo "SAMPLE_DONE: $CURRENT_COUNT/10 samples"
        # Show DB details
        for f in /tmp/reasoning-memory/benchmark_dbs/*.db; do
            if [ -f "$f" ]; then
                name=$(basename "$f")
                mems=$(python3 -c "import sqlite3; db=sqlite3.connect('$f'); print(db.execute('SELECT COUNT(*) FROM memories').fetchone()[0]); db.close()" 2>/dev/null)
                echo "  $name: $mems memories"
            fi
        done
        LAST_COUNT=$CURRENT_COUNT
    fi
    
    sleep 15
done

# Print final results
echo "=== FINAL RESULTS ==="
grep -v "WARNING" /tmp/reasoning-memory/benchmark_full.log | tail -30
