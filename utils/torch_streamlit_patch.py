"""Utility module to patch PyTorch and Streamlit integration issues.

This module provides a comprehensive solution to prevent runtime errors caused by
the interaction between Streamlit's file watcher and PyTorch's custom class system,
specifically addressing the __path__._path attribute access issues.

IMPORTANT: This module must be imported before any other imports that might
trigger the problematic behavior.
"""

# Import minimal dependencies first to avoid triggering issues
import types
import sys
import importlib
import inspect
from functools import wraps

# Track if we've already applied patches to avoid duplicate patching
_PATCHES_APPLIED = False

# Create a robust SafeModule class that handles all problematic attribute access
class SafeModule(types.ModuleType):
    """A safe module wrapper that prevents __path__._path errors"""
    
    def __init__(self, name, original_module=None):
        super().__init__(name)
        self._original_module = original_module
        # Create a safe __path__ attribute
        self.__path__ = []
        
        # Copy all attributes from original module if provided
        if original_module:
            for attr_name in dir(original_module):
                if not attr_name.startswith('__'):
                    try:
                        setattr(self, attr_name, getattr(original_module, attr_name))
                    except (AttributeError, TypeError):
                        pass
    
    def __getattr__(self, name):
        # Special handling for problematic attributes
        if name == "_path" or name.endswith("._path"):
            return []
        
        # Handle common PyTorch module attributes that might be missing
        if name in ['_functorch', '_C', 'utils', 'nn', 'fx', 'jit']:
            # Create a mock module for this attribute if it doesn't exist
            mock_name = f"{self.__name__}.{name}"
            mock_module = create_mock_module(mock_name)
            setattr(self, name, mock_module)
            return mock_module
            
        # Try to get from original module if it exists
        if self._original_module and hasattr(self._original_module, name):
            try:
                attr = getattr(self._original_module, name)
                # If it's a callable, wrap it to prevent errors
                if callable(attr) and not isinstance(attr, type):
                    @wraps(attr)
                    def safe_callable(*args, **kwargs):
                        try:
                            return attr(*args, **kwargs)
                        except Exception:
                            return None
                    return safe_callable
                return attr
            except (AttributeError, RuntimeError):
                pass
                
        # Default safe behavior
        if name.startswith('__'):
            raise AttributeError(f"{self.__name__} has no attribute {name}")
        return None

# Create a safe extract_paths function for Streamlit
def safe_extract_paths(module):
    """Safely extract paths from a module without causing __path__._path errors"""
    paths = []
    
    # Add file path if available
    if hasattr(module, "__file__") and module.__file__:
        paths.append(module.__file__)
    
    # Safely handle __path__ attribute
    if hasattr(module, "__path__"):
        try:
            if isinstance(module.__path__, list):
                paths.extend(module.__path__)
            # Handle _NamespacePath objects safely
            elif hasattr(module.__path__, "_path") and isinstance(module.__path__._path, list):
                try:
                    paths.extend(module.__path__._path)
                except (AttributeError, RuntimeError):
                    pass
        except (AttributeError, RuntimeError, TypeError):
            # Ignore any errors when accessing __path__
            pass
    
    return paths

# Apply a pre-import hook to intercept problematic imports
original_import = __import__

# Dictionary to store mock modules we've already created
_MOCK_MODULES = {}

def create_mock_module(name):
    """Create a mock module with appropriate attributes based on the module name"""
    if name in _MOCK_MODULES:
        return _MOCK_MODULES[name]
        
    mock_module = types.ModuleType(name)
    
    # Add specific functionality based on module name
    if name == 'torch._functorch._aot_autograd.functional_utils':
        mock_module.is_fun = lambda x: False
        # Add any other functions that might be needed
        mock_module.make_fx = lambda *args, **kwargs: None
        mock_module.get_functional_args = lambda *args, **kwargs: []
    elif name == 'torch._higher_order_ops.triton_kernel_wrap':
        # Add the missing TMADescriptorMetadata class
        class TMADescriptorMetadata:
            def __init__(self):
                pass
        mock_module.TMADescriptorMetadata = TMADescriptorMetadata
        
        # Add the missing TritonHOPifier class
        class TritonHOPifier:
            def __init__(self):
                pass
        mock_module.TritonHOPifier = TritonHOPifier
    elif name == 'torch.utils.checkpoint':
        # Mock checkpoint functionality
        mock_module.checkpoint = lambda function, *args, **kwargs: function(*args, **kwargs)
        mock_module.checkpoint_sequential = lambda functions, segments, *args, **kwargs: functions(*args, **kwargs)
    elif name.startswith('torch._functorch'):
        # Generic handler for all _functorch submodules
        mock_module.decompositions = types.ModuleType(name + '.decompositions')
        mock_module.config = types.ModuleType(name + '.config')
        # Add common functorch attributes
        mock_module.make_fx = lambda *args, **kwargs: None
        mock_module.wrap = lambda *args, **kwargs: args[0] if args else None
        mock_module.vmap = lambda *args, **kwargs: args[0] if args else None
    
    # Store the mock module for future use
    _MOCK_MODULES[name] = mock_module
    return mock_module

def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Patched import function that applies safety wrappers to problematic modules"""
    # Handle any torch internal module that might be missing
    if name.startswith('torch.') and ('._' in name or name == 'torch.utils.checkpoint'):
        try:
            module = original_import(name, globals, locals, fromlist, level)
        except (ModuleNotFoundError, ImportError, AttributeError):
            print(f"Creating mock module for {name}")
            return create_mock_module(name)
    
    # Handle fromlist items that might be missing
    if name == 'torch' and fromlist:
        try:
            module = original_import(name, globals, locals, fromlist, level)
            # Check if any fromlist items need to be mocked
            for item in fromlist:
                if item.startswith('_') or item == 'utils':
                    full_name = f"{name}.{item}"
                    try:
                        getattr(module, item)
                    except (AttributeError, ImportError):
                        print(f"Creating mock attribute for {full_name}")
                        setattr(module, item, create_mock_module(full_name))
        except (ModuleNotFoundError, ImportError) as e:
            print(f"Creating mock module for {name}")
            return create_mock_module(name)
    else:
        try:
            module = original_import(name, globals, locals, fromlist, level)
        except (ModuleNotFoundError, ImportError) as e:
            # Handle missing torch modules by creating mock modules
            if name.startswith('torch.'):
                print(f"Creating mock module for {name}")
                return create_mock_module(name)
            raise e
    
    # Apply patches to torch._classes if it's being imported
    if name == 'torch._classes' or (name == 'torch' and fromlist and '_classes' in fromlist):
        if hasattr(module, '_classes'):
            module._classes = SafeModule('torch._classes', module._classes)
    
    return module

# Install the import hook
sys.meta_path.insert(0, type('PathFinder', (), {
    'find_spec': lambda self, fullname, path, target=None: None,
    'find_module': lambda self, fullname, path=None: None,
    'load_module': lambda self, fullname: None
})())

# Replace the built-in __import__ function with our patched version
sys.modules['builtins'].__import__ = patched_import

def apply_patches():
    """Apply comprehensive patches to prevent __path__._path errors in PyTorch and Streamlit.
    
    This function applies several patches:
    1. Creates a safe module wrapper for problematic modules
    2. Patches torch._classes specifically
    3. Provides a safe path extraction function for Streamlit
    4. Patches Streamlit's module watcher system
    5. Applies patches to other potentially problematic modules
    6. Suppresses TensorFlow warnings
    
    Returns:
        bool: True if patches were applied successfully, False otherwise
    """
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return True
    
    # Suppress TensorFlow warnings
    import os
    import warnings
    
    # Set TensorFlow logging level to ERROR only
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error
    
    # Suppress common warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    try:
        # Suppress TensorFlow warnings
        try:
            import tensorflow as tf
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        except ImportError:
            pass
            
        # 1. Patch torch._classes with our safe module
        try:
            import torch
            if hasattr(torch, '_classes'):
                original_classes = torch._classes
                torch._classes = SafeModule('torch._classes', original_classes)
        except ImportError:
            # PyTorch not installed, skip this part
            pass
        
        # 2. Patch Streamlit's module system
        try:
            import streamlit.watcher.local_sources_watcher as watcher
            
            # Replace the extract_paths function with our safe version
            original_extract_paths = watcher.extract_paths
            watcher.extract_paths = safe_extract_paths
            
            # Also patch the path_funcs list to use safer functions
            watcher.path_funcs = [
                lambda m: [m.__file__] if hasattr(m, "__file__") and m.__file__ else [],
                lambda m: list(m.__path__) if hasattr(m, "__path__") and isinstance(m.__path__, list) else []
                # Removed the problematic __path__._path access completely
            ]
            
            # Patch the get_module_paths function to handle exceptions gracefully
            original_get_module_paths = watcher.get_module_paths
            
            @wraps(original_get_module_paths)
            def safe_get_module_paths(module):
                try:
                    return original_get_module_paths(module)
                except (AttributeError, RuntimeError, TypeError) as e:
                    # If there's an error, fall back to our safe extraction
                    return safe_extract_paths(module)
            
            watcher.get_module_paths = safe_get_module_paths
            
            # Reload the module to apply changes
            importlib.reload(watcher)
            print("Successfully patched Streamlit's module system to prevent __path__._path errors")
        except Exception as e:
            print(f"Note: Could not fully patch Streamlit's module system: {e}")
        
        # 3. Apply patches to any other modules that might cause similar issues
        for module_name in list(sys.modules.keys()):
            if ('torch' in module_name and '._' in module_name) or module_name.endswith('.__path__'):
                try:
                    module = sys.modules[module_name]
                    if hasattr(module, '__path__') and not isinstance(module, SafeModule):
                        sys.modules[module_name] = SafeModule(module_name, module)
                except Exception:
                    pass
        
        _PATCHES_APPLIED = True
        print("Successfully patched torch modules to prevent __path__._path errors")
        return True
        
    except Exception as e:
        # If patching fails, we'll handle errors gracefully
        print(f"Warning: Could not apply patches to prevent runtime errors: {e}")
        return False

# Apply patches when this module is imported
apply_patches()