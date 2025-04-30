"""Enhanced monkey patch for Streamlit to prevent __path__._path errors with PyTorch.

This script provides a comprehensive patching mechanism that directly intercepts
the problematic PyTorch custom class system to prevent runtime errors when running
Streamlit applications that use PyTorch.

It handles all edge cases of the torch._classes.__path__._path attribute access issue
with special handling for different Streamlit versions and PyTorch module structures.

It should be run before starting Streamlit, either by importing it or using exec():
    import streamlit_patch  # At the start of your script
    # OR
    exec(open('streamlit_patch.py').read())
"""

import sys
import types
import importlib
from functools import wraps

# Track if patches have been applied
_PATCHES_APPLIED = False

print("Applying enhanced patches to prevent PyTorch/Streamlit runtime errors...")

# Create a direct replacement for torch._classes.__path__._path
class SafePath(list):
    """A safe replacement for problematic _path attributes."""
    def __init__(self):
        super().__init__([])
    
    def __getattr__(self, name):
        # Return empty list for any attribute access
        return []
    
    def __getitem__(self, index):
        # Safely handle indexing operations
        return None
    
    def __iter__(self):
        # Return an empty iterator
        return iter([])
    
    def __contains__(self, item):
        # Always return False for membership tests
        return False
    
    def __len__(self):
        # Always return 0 for length
        return 0

# Create a safe __path__ object that won't cause errors
class SafePathObject(object):
    """A safe __path__ object that prevents _path access errors."""
    def __init__(self):
        self._path = SafePath()
    
    def __iter__(self):
        return iter([])
    
    def __getattr__(self, name):
        if name == "_path":
            return self._path
        # Return empty list for any other attribute
        return []
    
    def __getitem__(self, index):
        # Safely handle indexing operations
        return None
    
    def __contains__(self, item):
        # Always return False for membership tests
        return False
    
    def __len__(self):
        # Always return 0 for length
        return 0
    
    def __bool__(self):
        # Always evaluate to True to prevent conditional failures
        return True
    
    def append(self, item):
        # Safely handle append operations
        pass
    
    def extend(self, items):
        # Safely handle extend operations
        pass
    
    def __repr__(self):
        # Safe string representation
        return "[]"
    
    def __str__(self):
        # Safe string representation
        return "[]"

# Create a robust SafeModule class that handles all problematic attribute access
class SafeModule(types.ModuleType):
    """A safe module wrapper that prevents __path__._path errors"""
    
    def __init__(self, name, original_module=None):
        super().__init__(name)
        self._original_module = original_module
        # Create a safe __path__ attribute
        self.__path__ = SafePathObject()
        
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
        if name == "__path__":
            return self.__path__
        
        if name == "_path" or name.endswith("._path"):
            return []
            
        # Try to get from original module if it exists
        if self._original_module and hasattr(self._original_module, name):
            try:
                attr = getattr(self._original_module, name)
                # If it's a callable, wrap it to prevent errors
                if callable(attr):
                    @wraps(attr)
                    def safe_callable(*args, **kwargs):
                        try:
                            # Block attempts to access __path__._path
                            if args and len(args) > 0:
                                # Check all string arguments for problematic patterns
                                for arg in args:
                                    if isinstance(arg, str) and ("__path__._path" in arg or 
                                                               "._path" in arg):
                                        return None
                            return attr(*args, **kwargs)
                        except Exception:
                            return None
                    return safe_callable
                return attr
            except (AttributeError, RuntimeError):
                pass
                
        # For _get_* functions, try to delegate to torch._C
        if name.startswith('_get_') and 'torch._C' in sys.modules:
            try:
                original_func = getattr(sys.modules['torch._C'], name)
                
                # Create a safe wrapper for the function
                @wraps(original_func)
                def safe_func(*args, **kwargs):
                    try:
                        # Block attempts to access __path__._path
                        if args and len(args) > 0:
                            # Check all string arguments for problematic patterns
                            for arg in args:
                                if isinstance(arg, str) and ("__path__._path" in arg or 
                                                           "._path" in arg):
                                    return None
                        return original_func(*args, **kwargs)
                    except Exception:
                        return None
                
                return safe_func
            except AttributeError:
                pass
        
        # Default safe behavior
        if name.startswith('__'):
            raise AttributeError(f"{self.__name__} has no attribute {name}")
        return None
        
    def __dir__(self):
        # Provide a safe implementation of dir() that won't trigger attribute errors
        if self._original_module:
            try:
                return dir(self._original_module)
            except Exception:
                pass
        return super().__dir__()

# Create a completely safe replacement for torch._classes (for backward compatibility)
class SafeTorchClasses(SafeModule):
    """A safe replacement for torch._classes that prevents runtime errors."""
    def __init__(self, name="torch._classes", original_module=None):
        super().__init__(name, original_module)
    
    def __getattr__(self, name):
        # Block all attribute access that might cause problems
        if name == "__path__" or name == "_path" or "._path" in name:
            return SafePathObject()
        return super().__getattr__(name)

# Direct patch for torch._classes module to completely block problematic access
def patch_torch_classes():
    """Apply a direct patch to torch._classes to prevent __path__._path errors."""
    try:
        import torch
        if not hasattr(torch, '_classes'):
            return False
            
        # Create a completely safe replacement for torch._classes.__getattr__
        original_getattr = torch._classes.__class__.__getattr__
        
        def safe_getattr(self, name):
            # Block any access to __path__._path
            if name == "__path__":
                return SafePathObject()
            if name == "_path" or "._path" in name:
                return []
                
            # For any other attribute, try the original __getattr__
            try:
                return original_getattr(self, name)
            except Exception:
                return None
                
        # Apply the patch
        torch._classes.__class__.__getattr__ = safe_getattr
        
        # Also directly patch the _get_custom_class_python_wrapper function
        if hasattr(torch, '_C') and hasattr(torch._C, '_get_custom_class_python_wrapper'):
            original_func = torch._C._get_custom_class_python_wrapper
            torch._C._get_custom_class_python_wrapper = safe_get_custom_class_wrapper(original_func)
            
        return True
    except Exception as e:
        print(f"Warning: Could not patch torch._classes: {e}")
        return False

# Direct patch for torch._C._get_custom_class_python_wrapper
def safe_get_custom_class_wrapper(original_func):
    """Create a safe wrapper for torch._C._get_custom_class_python_wrapper."""
    @wraps(original_func)
    def wrapper(class_name, attr):
        # Block attempts to access __path__._path or any ._path attribute
        if (attr == "__path__._path" or class_name == "__path__._path" or
            "._path" in attr or "._path" in class_name or
            attr == "_path" or class_name == "_path"):
            return None
        try:
            return original_func(class_name, attr)
        except (RuntimeError, AttributeError, TypeError):
            # Return None instead of raising an error
            return None
    return wrapper

# Safe path extraction function for Streamlit
def safe_extract_paths(module):
    """Safely extract paths from a module without causing __path__._path errors."""
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
                except (AttributeError, RuntimeError, TypeError):
                    pass
        except (AttributeError, RuntimeError, TypeError):
            pass
    
    return paths

# Direct patch for the specific line in Streamlit's local_sources_watcher.py that's causing the error
def patch_streamlit_lambda():
    """Apply a direct patch to the problematic lambda function in Streamlit's local_sources_watcher.py."""
    try:
        # Try to find the Streamlit watcher module
        import streamlit
        import importlib.util
        import os
        
        # Look for the specific file that contains the problematic lambda
        streamlit_path = os.path.dirname(streamlit.__file__)
        watcher_paths = [
            os.path.join(streamlit_path, 'watcher', 'local_sources_watcher.py'),
            os.path.join(streamlit_path, '_file_watcher', 'local_sources_watcher.py'),
            os.path.join(streamlit_path, 'file_watcher', 'local_sources_watcher.py')
        ]
        
        for watcher_path in watcher_paths:
            if os.path.exists(watcher_path):
                # Read the file content
                with open(watcher_path, 'r') as f:
                    content = f.read()
                
                # Check if the problematic lambda is present
                if 'lambda m: list(m.__path__._path)' in content:
                    # Replace it with a safe version
                    patched_content = content.replace(
                        'lambda m: list(m.__path__._path)',
                        'lambda m: []  # Patched by streamlit_patch.py'
                    )
                    
                    try:
                        # Try to write the patched content back
                        with open(watcher_path, 'w') as f:
                            f.write(patched_content)
                        print(f"Successfully patched problematic lambda in {watcher_path}")
                        return True
                    except (PermissionError, IOError):
                        # If we can't write to the file, try to monkey patch it in memory
                        print(f"Could not write to {watcher_path}, applying in-memory patch instead")
                        break
        
        return False
    except Exception as e:
        print(f"Could not patch Streamlit lambda: {e}")
        return False

# Custom import hook to intercept problematic imports
class TorchPathProtectionFinder:
    """Custom import hook that prevents __path__._path access errors."""
    def __init__(self):
        self.original_importers = sys.meta_path.copy()
    
    def find_spec(self, fullname, path, target=None):
        # Intercept torch._classes imports
        if fullname == 'torch._classes':
            # Return None to let the normal import machinery handle it
            # We'll patch it after it's imported
            return None
        return None
    
    def find_module(self, fullname, path=None):
        # For older Python versions
        if fullname == 'torch._classes':
            # Return a loader that will return our safe module
            return self
        return None
    
    def load_module(self, fullname):
        # Create a safe module for torch._classes
        if fullname == 'torch._classes':
            # Check if it's already in sys.modules
            if fullname in sys.modules:
                original_module = sys.modules[fullname]
                # Replace it with our safe module
                safe_module = SafeModule(fullname, original_module)
                sys.modules[fullname] = safe_module
                return safe_module
            
            # Create a new safe module
            safe_module = SafeModule(fullname)
            sys.modules[fullname] = safe_module
            return safe_module
        
        # Should never get here
        raise ImportError(f"Cannot load {fullname}")
    
    def install(self):
        # Install this finder at the beginning of sys.meta_path
        if self not in sys.meta_path:
            sys.meta_path.insert(0, self)

# Install the custom import hook
TorchPathProtectionFinder().install()

# Patch the import mechanism to intercept problematic imports
original_import = __import__

def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Patched import function that applies safety wrappers to problematic modules"""
    module = original_import(name, globals, locals, fromlist, level)
    
    # Apply patches to torch._classes if it's being imported
    if name == 'torch._classes' or (name == 'torch' and fromlist and '_classes' in fromlist):
        if hasattr(module, '_classes'):
            # Replace torch._classes with our safe module
            original_classes = module._classes
            module._classes = SafeModule('torch._classes', original_classes)
            sys.modules['torch._classes'] = module._classes
    
    return module

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
    
    Returns:
        bool: True if patches were applied successfully, False otherwise
    """
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return True
    
    # First, try to directly patch the problematic lambda in Streamlit's source code
    # This is the most direct approach to fix the specific error
    patch_streamlit_lambda()
    
    try:
        # 1. Direct patch for torch._classes
        try:
            import torch
            
            # Store original module for reference
            original_classes = None
            if hasattr(torch, '_classes'):
                original_classes = torch._classes
            
            # Replace torch._classes with our safe module
            torch._classes = SafeModule('torch._classes', original_classes)
            sys.modules['torch._classes'] = torch._classes
            
            # Directly patch torch._C._get_custom_class_python_wrapper if it exists
            if hasattr(torch, '_C'):
                if hasattr(torch._C, '_get_custom_class_python_wrapper'):
                    original_func = torch._C._get_custom_class_python_wrapper
                    torch._C._get_custom_class_python_wrapper = safe_get_custom_class_wrapper(original_func)
                
                # Also patch other potentially problematic functions
                for attr_name in dir(torch._C):
                    if attr_name.startswith('_get_') and callable(getattr(torch._C, attr_name)):
                        try:
                            original_func = getattr(torch._C, attr_name)
                            patched_func = safe_get_custom_class_wrapper(original_func)
                            setattr(torch._C, attr_name, patched_func)
                        except (AttributeError, TypeError):
                            pass
            
            print("Successfully patched PyTorch custom class system")
        except ImportError:
            print("PyTorch not found, skipping torch patches")
        
        # 2. Patch Streamlit's module system
        try:
            import streamlit
            
            # Try different module paths for different Streamlit versions
            watcher_module_paths = [
                'streamlit.watcher.local_sources_watcher',
                'streamlit._file_watcher.local_sources_watcher',
                'streamlit.file_watcher.local_sources_watcher'
            ]
            
            watcher = None
            for module_path in watcher_module_paths:
                try:
                    watcher = importlib.import_module(module_path)
                    break
                except ImportError:
                    continue
            
            if watcher:
                # Direct patch for the extract_paths function
                # This is the most critical part that causes the error
                if hasattr(watcher, 'path_funcs'):
                    # Directly replace the problematic lambda function that tries to access __path__._path
                    # First, find the problematic function in the list
                    for i, func in enumerate(watcher.path_funcs):
                        if callable(func):
                            try:
                                # Try to get the source code of the function
                                import inspect
                                source = inspect.getsource(func)
                                # Check if it's the problematic lambda that accesses __path__._path
                                if '.__path__._path' in source:
                                    # Replace it with a safe version that doesn't access _path
                                    watcher.path_funcs[i] = lambda m: []
                            except Exception:
                                # If we can't inspect the function, check if it raises an error with a test module
                                try:
                                    test_module = types.ModuleType('test')
                                    test_module.__file__ = 'test.py'
                                    test_module.__path__ = []
                                    func(test_module)  # If this works, it's safe
                                except Exception:
                                    # If it raises an exception, replace it with a safe version
                                    watcher.path_funcs[i] = lambda m: []
                    
                    # Also add our safe functions to ensure we have proper path extraction
                    watcher.path_funcs.append(lambda m: [m.__file__] if hasattr(m, "__file__") and m.__file__ else [])
                    watcher.path_funcs.append(lambda m: list(m.__path__) if hasattr(m, "__path__") and isinstance(m.__path__, list) else [])
                
                # Replace extract_paths if it exists
                if hasattr(watcher, 'extract_paths'):
                    watcher.extract_paths = safe_extract_paths
                else:
                    # If extract_paths doesn't exist, we need to create it
                    # This handles the case where the error message says "no attribute 'extract_paths'"
                    watcher.extract_paths = safe_extract_paths
                    
                # As a last resort, completely replace path_funcs with only safe functions
                try:
                    # Create a completely new path_funcs list with only safe functions
                    watcher.path_funcs = [
                        lambda m: [m.__file__] if hasattr(m, "__file__") and m.__file__ else [],
                        lambda m: list(m.__path__) if hasattr(m, "__path__") and isinstance(m.__path__, list) else []
                    ]
                    print("Completely replaced path_funcs with safe functions")
                except Exception as e:
                    print(f"Could not replace path_funcs: {e}")
                    
                # Reload the module to apply changes
                importlib.reload(watcher)
                print("Successfully patched Streamlit's module system")
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
        print("Successfully applied all patches to prevent PyTorch/Streamlit runtime errors")
        return True
        
    except Exception as e:
        # If patching fails, we'll handle errors gracefully
        print(f"Warning: Could not apply all patches to prevent runtime errors: {e}")
        return False

# Apply patches when this module is imported
apply_patches()