"""
Executes a Jupyter notebook and saves all outputs inline without requiring ZMQ kernel sockets.
"""
import sys
import os
import io
import contextlib
import base64
import nbformat as nbf
import matplotlib.pyplot as plt

def execute_notebook(nb_path: str, out_path: str = None):
    if out_path is None:
        out_path = nb_path
    
    print(f"Executing notebook: {nb_path}")
    nb = nbf.read(nb_path, as_version=4)
    
    # Execution namespace
    ns = {
        '__name__': '__main__',
        'display': lambda x: print(x)
    }
    
    # Custom display that captures DataFrames
    def custom_display(obj):
        if hasattr(obj, 'to_html'):
            captured_displays.append(('text/html', obj.to_html()))
        elif isinstance(obj, str):
            captured_displays.append(('text/plain', obj))
        else:
            captured_displays.append(('text/plain', repr(obj)))

    ns['display'] = custom_display

    exec_count = 1
    for cell_idx, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            captured_displays = []
            code = cell.source
            # Strip IPython magics like %matplotlib inline
            cleaned_lines = []
            for line in code.split('\n'):
                if line.strip().startswith('%') or line.strip().startswith('!'):
                    continue
                cleaned_lines.append(line)
            executable_code = '\n'.join(cleaned_lines)
            
            stdout_io = io.StringIO()
            stderr_io = io.StringIO()
            
            plt.close('all')
            try:
                with contextlib.redirect_stdout(stdout_io), contextlib.redirect_stderr(stderr_io):
                    exec(executable_code, ns)
                
                cell.execution_count = exec_count
                exec_count += 1
                cell.outputs = []
                
                # Check for standard text output
                stdout_text = stdout_io.getvalue()
                if stdout_text:
                    cell.outputs.append(nbf.v4.new_output('stream', name='stdout', text=stdout_text))
                
                # Check for rich displays (DataFrames)
                for mime, data in captured_displays:
                    cell.outputs.append(nbf.v4.new_output('display_data', data={mime: data}))
                
                # Check for Matplotlib figures
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    cell.outputs.append(nbf.v4.new_output(
                        'display_data',
                        data={'image/png': img_base64, 'text/plain': '<Figure size ...>'}
                    ))
                plt.close('all')
                
            except Exception as e:
                print(f"Error in cell {cell_idx}: {e}")
                cell.outputs.append(nbf.v4.new_output(
                    'error',
                    ename=type(e).__name__,
                    evalue=str(e),
                    traceback=[f"Error: {e}"]
                ))

    nbf.write(nb, out_path)
    print(f"Notebook successfully executed and saved with outputs to: {out_path}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "notebooks/02_preprocessing_week6.ipynb"
    execute_notebook(target)
